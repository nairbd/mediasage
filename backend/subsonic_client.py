"""Subsonic / OpenSubsonic media server client (covers Navidrome, Airsonic, Gonic, etc.)."""

import hashlib
import logging
import secrets
from typing import Any

import httpx

from backend.media_client import BaseMediaClient, is_live_track
from backend.models import PlexPlaylistInfo, Track

logger = logging.getLogger(__name__)

# Singleton instance
_subsonic_client: "SubsonicClient | None" = None

# API constants
API_VERSION = "1.16.1"
CLIENT_NAME = "MediaSage"
PAGE_SIZE = 500  # search3 max per page


def _decade_to_years(decade: str) -> list[int]:
    """Convert a decade string like '1980s' to a list of years [1980..1989]."""
    decade = decade.strip().rstrip("s")
    try:
        start = int(decade)
        return list(range(start, start + 10))
    except ValueError:
        return []


class SubsonicClient(BaseMediaClient):
    """Client for Subsonic / OpenSubsonic-compatible servers (Navidrome, Airsonic, etc.)."""

    def __init__(self, url: str, username: str, password: str, music_library: str = ""):
        self.url = url.rstrip("/")
        self.username = username
        self.password = password
        # music_library is informational here; Subsonic exposes folders via getMusicFolders
        # but most servers (Navidrome) only have one folder, so we don't filter by it for now.
        self.music_library_name = music_library

        self._connected = False
        self._error: str | None = None
        self._server_type: str | None = None
        self._server_version: str | None = None

        self._connect()

    def _auth_params(self) -> dict[str, str]:
        """Generate token+salt auth params (avoids sending plain password)."""
        salt = secrets.token_hex(8)
        token = hashlib.md5((self.password + salt).encode("utf-8")).hexdigest()
        return {
            "u": self.username,
            "t": token,
            "s": salt,
            "v": API_VERSION,
            "c": CLIENT_NAME,
            "f": "json",
        }

    def _get(self, client: httpx.Client, endpoint: str, **params) -> dict[str, Any]:
        """GET a Subsonic endpoint and return the parsed `subsonic-response` payload.

        Raises httpx.HTTPStatusError on transport error and RuntimeError on Subsonic-level
        error (status='failed').
        """
        merged = {**self._auth_params(), **params}
        resp = client.get(f"{self.url}/rest/{endpoint}", params=merged)
        resp.raise_for_status()
        body = resp.json().get("subsonic-response", {})
        if body.get("status") == "failed":
            err = body.get("error", {})
            raise RuntimeError(f"Subsonic error {err.get('code')}: {err.get('message')}")
        return body

    def _connect(self) -> None:
        """Ping the server and record version info."""
        if not self.url or not self.username or not self.password:
            self._error = "Subsonic URL, username, and password are required"
            return

        try:
            with httpx.Client(timeout=15.0) as client:
                body = self._get(client, "ping.view")
                self._server_type = body.get("type")
                self._server_version = body.get("serverVersion") or body.get("version")
                self._connected = True
                self._error = None
                logger.info(
                    "Connected to Subsonic: type=%s version=%s",
                    self._server_type,
                    self._server_version,
                )
        except httpx.ConnectError:
            self._error = f"Cannot connect to Subsonic server at {self.url}"
            self._connected = False
        except httpx.HTTPStatusError as e:
            self._error = f"Subsonic returned HTTP {e.response.status_code}"
            self._connected = False
        except RuntimeError as e:
            # Subsonic-level error (bad credentials, etc.)
            msg = str(e)
            if "code: 40" in msg or "Wrong username" in msg:
                self._error = "Invalid Subsonic credentials"
            else:
                self._error = msg
            self._connected = False
        except Exception as e:
            self._error = f"Subsonic connection error: {str(e)}"
            self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def get_error(self) -> str | None:
        return self._error

    def get_music_libraries(self) -> list[str]:
        """Get list of music folder names from Subsonic."""
        if not self._connected:
            return []
        try:
            with httpx.Client(timeout=10.0) as client:
                body = self._get(client, "getMusicFolders.view")
                folders = body.get("musicFolders", {}).get("musicFolder", [])
                return [f["name"] for f in folders]
        except Exception as e:
            logger.warning("Failed to get Subsonic music folders: %s", e)
            return []

    def _song_to_track(self, song: dict) -> Track:
        """Convert a Subsonic song dict to a Track model."""
        item_id = song["id"]
        duration_s = song.get("duration") or 0
        # Prefer richer artist fields where available (OpenSubsonic).
        artist = (
            song.get("displayAlbumArtist")
            or song.get("albumArtist")
            or song.get("displayArtist")
            or song.get("artist")
            or ""
        )
        # Genres can come as string or list of {name: ...}
        genres: list[str] = []
        if song.get("genres"):
            genres = [g.get("name", "") for g in song["genres"] if g.get("name")]
        elif song.get("genre"):
            genres = [song["genre"]]

        return Track(
            rating_key=item_id,
            title=song.get("title", ""),
            artist=artist,
            album=song.get("album", ""),
            duration_ms=duration_s * 1000,
            year=song.get("year"),
            genres=genres,
            art_url=f"/api/art/{item_id}",
        )

    def get_library_stats(self) -> dict[str, Any]:
        """Get statistics about the Subsonic music library."""
        if not self._connected:
            return {"total_tracks": 0, "genres": [], "decades": []}

        try:
            with httpx.Client(timeout=60.0) as client:
                # Genres come with songCount baked in
                body = self._get(client, "getGenres.view")
                genre_items = body.get("genres", {}).get("genre", [])
                total_tracks = sum(g.get("songCount", 0) for g in genre_items)
                genres = sorted(
                    [{"name": g["value"], "count": g.get("songCount")} for g in genre_items],
                    key=lambda x: x["name"],
                )

                # Decades: walk all albums (smaller than walking all songs) and bucket by year.
                decade_set: set[str] = set()
                offset = 0
                album_page = 500
                while True:
                    body = self._get(
                        client,
                        "getAlbumList2.view",
                        type="alphabeticalByName",
                        size=album_page,
                        offset=offset,
                    )
                    albums = body.get("albumList2", {}).get("album", [])
                    if not albums:
                        break
                    for album in albums:
                        year = album.get("year")
                        if year:
                            decade_set.add(f"{(year // 10) * 10}s")
                    if len(albums) < album_page:
                        break
                    offset += album_page

                decades = sorted(
                    [{"name": d, "count": None} for d in decade_set],
                    key=lambda x: x["name"],
                )

                return {
                    "total_tracks": total_tracks,
                    "genres": genres,
                    "decades": decades,
                }
        except Exception as e:
            logger.exception("Failed to get Subsonic library stats: %s", e)
            return {"total_tracks": 0, "genres": [], "decades": [], "error": str(e)}

    def get_all_tracks(self) -> list[Track]:
        """Get all tracks via search3 with empty query (Navidrome convention)."""
        if not self._connected:
            return []

        tracks: list[Track] = []
        offset = 0

        with httpx.Client(timeout=300.0) as client:
            while True:
                body = self._get(
                    client,
                    "search3.view",
                    query="",
                    songCount=PAGE_SIZE,
                    songOffset=offset,
                    albumCount=0,
                    artistCount=0,
                )
                songs = body.get("searchResult3", {}).get("song", [])
                if not songs:
                    break
                tracks.extend(self._song_to_track(s) for s in songs)
                if len(songs) < PAGE_SIZE:
                    break
                offset += PAGE_SIZE

        return tracks

    def get_all_albums_metadata(self) -> dict[str, dict[str, Any]]:
        """Walk all albums via getAlbumList2 and return mapping of album_id -> metadata."""
        if not self._connected:
            return {}

        try:
            result: dict[str, dict[str, Any]] = {}
            offset = 0
            page = 500

            with httpx.Client(timeout=300.0) as client:
                while True:
                    body = self._get(
                        client,
                        "getAlbumList2.view",
                        type="alphabeticalByName",
                        size=page,
                        offset=offset,
                    )
                    albums = body.get("albumList2", {}).get("album", [])
                    if not albums:
                        break
                    for album in albums:
                        # Collect genres from both legacy 'genre' string and OpenSubsonic 'genres' list
                        genres: list[str] = []
                        if album.get("genres"):
                            genres = [g.get("name", "") for g in album["genres"] if g.get("name")]
                        elif album.get("genre"):
                            genres = [album["genre"]]
                        result[album["id"]] = {
                            "genres": genres,
                            "year": album.get("year"),
                        }
                    if len(albums) < page:
                        break
                    offset += page

            return result
        except Exception as e:
            logger.exception("Failed to get Subsonic album metadata: %s", e)
            return {}

    def _post_filter_tracks(
        self,
        tracks: list[Track],
        decades: list[str] | None,
        exclude_live: bool,
    ) -> list[Track]:
        """Apply decade + live filters in Python (server-side filters are limited)."""
        years_filter: set[int] | None = None
        if decades:
            ys: list[int] = []
            for d in decades:
                ys.extend(_decade_to_years(d))
            if ys:
                years_filter = set(ys)

        out: list[Track] = []
        for t in tracks:
            if years_filter is not None and (t.year is None or t.year not in years_filter):
                continue
            if exclude_live and is_live_track(t.title, t.album):
                continue
            out.append(t)
        return out

    def get_tracks_by_filters(
        self,
        genres: list[str] | None = None,
        decades: list[str] | None = None,
        exclude_live: bool = True,
        min_rating: int = 0,
        limit: int = 0,
    ) -> list[Track]:
        """Get tracks matching filter criteria from Subsonic.

        Strategy:
        - If genres given: paginate getSongsByGenre per genre (server-side filter), then
          filter decades/live in Python.
        - Otherwise: walk all tracks via search3 and filter in Python.
        - min_rating ignored (Subsonic ratings don't map cleanly across servers).
        """
        if not self._connected:
            return []

        try:
            tracks: list[Track] = []

            with httpx.Client(timeout=300.0) as client:
                if genres:
                    seen_ids: set[str] = set()
                    for genre in genres:
                        offset = 0
                        while True:
                            body = self._get(
                                client,
                                "getSongsByGenre.view",
                                genre=genre,
                                count=PAGE_SIZE,
                                offset=offset,
                            )
                            songs = body.get("songsByGenre", {}).get("song", [])
                            if not songs:
                                break
                            for s in songs:
                                if s["id"] in seen_ids:
                                    continue
                                seen_ids.add(s["id"])
                                tracks.append(self._song_to_track(s))
                            if len(songs) < PAGE_SIZE:
                                break
                            offset += PAGE_SIZE
                else:
                    offset = 0
                    while True:
                        body = self._get(
                            client,
                            "search3.view",
                            query="",
                            songCount=PAGE_SIZE,
                            songOffset=offset,
                            albumCount=0,
                            artistCount=0,
                        )
                        songs = body.get("searchResult3", {}).get("song", [])
                        if not songs:
                            break
                        tracks.extend(self._song_to_track(s) for s in songs)
                        if len(songs) < PAGE_SIZE:
                            break
                        offset += PAGE_SIZE

            tracks = self._post_filter_tracks(tracks, decades, exclude_live)

            if limit > 0:
                tracks = tracks[:limit]

            return tracks
        except Exception as e:
            logger.exception("Failed to get Subsonic filtered tracks: %s", e)
            return []

    def get_random_tracks(
        self,
        count: int,
        exclude_live: bool = True,
    ) -> list[Track]:
        """Get random tracks via getRandomSongs (capped at 500 per call)."""
        if not self._connected:
            return []

        try:
            # Over-fetch when filtering live, since the endpoint can't pre-filter
            target = count * 3 if exclude_live else count
            target = min(target, 500)  # Subsonic cap

            with httpx.Client(timeout=30.0) as client:
                body = self._get(client, "getRandomSongs.view", size=target)
                songs = body.get("randomSongs", {}).get("song", [])

                tracks: list[Track] = []
                for s in songs:
                    track = self._song_to_track(s)
                    if exclude_live and is_live_track(track.title, track.album):
                        continue
                    tracks.append(track)
                    if len(tracks) >= count:
                        break
                return tracks
        except Exception as e:
            logger.exception("Failed to get random Subsonic tracks: %s", e)
            return []

    def get_track_by_key(self, rating_key: str) -> Track | None:
        """Get a single track by its Subsonic song ID."""
        if not self._connected:
            return None

        try:
            with httpx.Client(timeout=10.0) as client:
                body = self._get(client, "getSong.view", id=rating_key)
                song = body.get("song")
                if not song:
                    return None
                return self._song_to_track(song)
        except Exception as e:
            logger.warning("Failed to get Subsonic track %s: %s", rating_key, e)
            return None

    def search_tracks(self, query: str) -> list[Track]:
        """Search for tracks by title or artist via search3."""
        if not self._connected:
            return []

        try:
            with httpx.Client(timeout=15.0) as client:
                body = self._get(
                    client,
                    "search3.view",
                    query=query,
                    songCount=50,
                    albumCount=0,
                    artistCount=0,
                )
                songs = body.get("searchResult3", {}).get("song", [])
                return [self._song_to_track(s) for s in songs]
        except Exception as e:
            logger.warning("Failed to search Subsonic tracks: %s", e)
            return []

    def count_tracks_by_filters(
        self,
        genres: list[str] | None = None,
        decades: list[str] | None = None,
        exclude_live: bool = True,
        min_rating: int = 0,
    ) -> int:
        """Count tracks matching filters.

        Subsonic has no native count endpoint with these filters. Fast path: when only
        genres are given (no decade/live filter), sum songCount from getGenres. Otherwise
        fall back to a full fetch.
        """
        if not self._connected:
            return 0

        # Fast path: genres only, no decade filter, no live filter
        if genres and not decades and not exclude_live:
            try:
                with httpx.Client(timeout=15.0) as client:
                    body = self._get(client, "getGenres.view")
                    items = body.get("genres", {}).get("genre", [])
                    by_name = {g["value"]: g.get("songCount", 0) for g in items}
                    return sum(by_name.get(g, 0) for g in genres)
            except Exception as e:
                logger.warning("Failed fast-path Subsonic count: %s", e)
                # Fall through to slow path

        tracks = self.get_tracks_by_filters(
            genres=genres,
            decades=decades,
            exclude_live=exclude_live,
            min_rating=min_rating,
            limit=0,
        )
        return len(tracks)

    def create_playlist(
        self,
        name: str,
        rating_keys: list[str],
        description: str = "",
    ) -> dict[str, Any]:
        """Create a playlist in Subsonic. Description set via updatePlaylist (OpenSubsonic 'comment')."""
        if not self._connected:
            return {"success": False, "error": "Not connected to Subsonic"}

        try:
            with httpx.Client(timeout=60.0) as client:
                # createPlaylist accepts repeated songId params
                params: list[tuple[str, str]] = list(self._auth_params().items())
                params.append(("name", name))
                for sid in rating_keys:
                    params.append(("songId", sid))
                resp = client.get(f"{self.url}/rest/createPlaylist.view", params=params)
                resp.raise_for_status()
                body = resp.json().get("subsonic-response", {})
                if body.get("status") == "failed":
                    err = body.get("error", {})
                    raise RuntimeError(f"Subsonic error {err.get('code')}: {err.get('message')}")

                playlist = body.get("playlist") or {}
                playlist_id = playlist.get("id")

                # Older Subsonic returns no body — fall back to looking up by name
                if not playlist_id:
                    body = self._get(client, "getPlaylists.view")
                    for p in body.get("playlists", {}).get("playlist", []):
                        if p.get("name") == name:
                            playlist_id = p["id"]
                            break

                if not playlist_id:
                    return {"success": False, "error": "Playlist created but no ID returned"}

                # Set description via updatePlaylist (Navidrome supports `comment`)
                if description:
                    try:
                        self._get(
                            client,
                            "updatePlaylist.view",
                            playlistId=playlist_id,
                            comment=description,
                        )
                    except Exception:
                        # Non-fatal
                        logger.warning("Failed to set playlist description on %s", playlist_id)

                return {
                    "success": True,
                    "playlist_id": playlist_id,
                    "playlist_url": None,
                    "tracks_added": len(rating_keys),
                    "tracks_skipped": 0,
                }
        except Exception as e:
            logger.exception("Failed to create Subsonic playlist '%s'", name)
            return {"success": False, "error": str(e)}

    def update_playlist(
        self,
        playlist_id: str,
        rating_keys: list[str],
        mode: str = "replace",
        description: str = "",
    ) -> dict[str, Any]:
        """Update a Subsonic playlist by replacing or appending tracks."""
        if not self._connected:
            return {"success": False, "error": "Not connected to Subsonic"}

        try:
            with httpx.Client(timeout=60.0) as client:
                if mode == "replace":
                    # Get current track count, then remove all by index 0..N-1
                    body = self._get(client, "getPlaylist.view", id=playlist_id)
                    playlist = body.get("playlist", {})
                    current_count = playlist.get("songCount", len(playlist.get("entry", [])))

                    if current_count > 0:
                        params: list[tuple[str, str]] = list(self._auth_params().items())
                        params.append(("playlistId", playlist_id))
                        for i in range(current_count):
                            params.append(("songIndexToRemove", str(i)))
                        resp = client.get(f"{self.url}/rest/updatePlaylist.view", params=params)
                        resp.raise_for_status()
                        body = resp.json().get("subsonic-response", {})
                        if body.get("status") == "failed":
                            err = body.get("error", {})
                            raise RuntimeError(
                                f"Subsonic error {err.get('code')}: {err.get('message')}"
                            )

                # Append new songs
                if rating_keys:
                    params = list(self._auth_params().items())
                    params.append(("playlistId", playlist_id))
                    for sid in rating_keys:
                        params.append(("songIdToAdd", sid))
                    if description:
                        params.append(("comment", description))
                    resp = client.get(f"{self.url}/rest/updatePlaylist.view", params=params)
                    resp.raise_for_status()
                    body = resp.json().get("subsonic-response", {})
                    if body.get("status") == "failed":
                        err = body.get("error", {})
                        raise RuntimeError(
                            f"Subsonic error {err.get('code')}: {err.get('message')}"
                        )
                elif description:
                    # Description-only update
                    self._get(
                        client,
                        "updatePlaylist.view",
                        playlistId=playlist_id,
                        comment=description,
                    )

                return {
                    "success": True,
                    "tracks_added": len(rating_keys),
                    "tracks_skipped": 0,
                    "duplicates_skipped": 0,
                    "playlist_url": None,
                }
        except Exception as e:
            logger.exception("Failed to update Subsonic playlist %s", playlist_id)
            return {"success": False, "error": str(e)}

    def get_playlists(self) -> list[PlexPlaylistInfo]:
        """Get all playlists from Subsonic (no media-type filter; Subsonic playlists are mixed)."""
        if not self._connected:
            return []

        try:
            with httpx.Client(timeout=15.0) as client:
                body = self._get(client, "getPlaylists.view")
                items = body.get("playlists", {}).get("playlist", [])
                result = [
                    PlexPlaylistInfo(
                        rating_key=p["id"],
                        title=p["name"],
                        track_count=p.get("songCount", 0),
                    )
                    for p in items
                ]
                return sorted(result, key=lambda p: p.title.lower())
        except Exception as e:
            logger.exception("Failed to get Subsonic playlists: %s", e)
            return []

    def get_art_url(self, item_id: str) -> str | None:
        """Get the cover-art URL for a Subsonic item.

        Returns a fully-authenticated URL the backend can proxy. The art id is the song or
        album ID itself — Subsonic resolves it.
        """
        if not self._connected or not self.url:
            return None
        params = self._auth_params()
        params["id"] = item_id
        # Build a query string with the auth params baked in
        from urllib.parse import urlencode
        return f"{self.url}/rest/getCoverArt.view?{urlencode(params)}"

    def get_machine_identifier(self) -> str | None:
        """Subsonic has no stable server-id endpoint; use type+version as a proxy."""
        if not self._server_type or not self._server_version:
            return None
        return f"{self._server_type}-{self._server_version}"

    def get_server_name(self) -> str | None:
        """Return server type label."""
        return self._server_type


def get_subsonic_client() -> SubsonicClient | None:
    """Get the global Subsonic client instance."""
    return _subsonic_client


def init_subsonic_client(
    url: str, username: str, password: str, music_library: str = ""
) -> SubsonicClient:
    """Initialize the global Subsonic client."""
    global _subsonic_client
    _subsonic_client = SubsonicClient(url, username, password, music_library)
    return _subsonic_client
