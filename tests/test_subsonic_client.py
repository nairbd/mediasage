"""Tests for Subsonic / OpenSubsonic client."""

import hashlib
from unittest.mock import MagicMock, patch

import httpx
import pytest

from backend.models import PlexPlaylistInfo, Track
from backend.subsonic_client import PAGE_SIZE, SubsonicClient


def _mock_response(payload: dict, status_code: int = 200) -> MagicMock:
    """Build a mock httpx.Response wrapping a Subsonic JSON envelope."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = {"subsonic-response": payload}
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"HTTP {status_code}", request=MagicMock(), response=resp
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


def _ok(**fields) -> dict:
    """Subsonic-response payload with status=ok and any extra fields."""
    return {"status": "ok", **fields}


def _failed(code: int, message: str) -> dict:
    """Subsonic-response payload with status=failed and error block."""
    return {"status": "failed", "error": {"code": code, "message": message}}


def _make_client_mock(get_responses):
    """Return a MagicMock standing in for httpx.Client used as a context manager.

    `get_responses` is a list (consumed in order) or a single response.
    """
    client = MagicMock()
    if isinstance(get_responses, list):
        client.get.side_effect = get_responses
    else:
        client.get.return_value = get_responses
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=client)
    cm.__exit__ = MagicMock(return_value=False)
    return cm, client


def _make_subsonic(connected: bool = True) -> SubsonicClient:
    """Construct a SubsonicClient bypassing _connect for tests that don't exercise it."""
    with patch.object(SubsonicClient, "_connect", lambda self: None):
        c = SubsonicClient("http://navi.test", "user", "pw", "Music")
    c._connected = connected
    c._server_type = "navidrome"
    c._server_version = "0.61.2"
    return c


# ---------------------------------------------------------------------------
# Auth params
# ---------------------------------------------------------------------------


class TestSubsonicAuthParams:
    """Tests for token+salt auth param generation."""

    def test_auth_params_includes_required_fields(self):
        client = _make_subsonic()
        params = client._auth_params()
        assert set(params) == {"u", "t", "s", "v", "c", "f"}
        assert params["u"] == "user"
        assert params["c"] == "MediaSage"
        assert params["f"] == "json"

    def test_token_is_md5_of_password_plus_salt(self):
        client = _make_subsonic()
        params = client._auth_params()
        expected = hashlib.md5(("pw" + params["s"]).encode("utf-8")).hexdigest()
        assert params["t"] == expected

    def test_salt_is_unique_per_call(self):
        client = _make_subsonic()
        salts = {client._auth_params()["s"] for _ in range(10)}
        assert len(salts) == 10


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------


class TestSubsonicConnection:
    """Tests for connection handling via ping.view."""

    def test_connect_with_valid_credentials(self):
        cm, _ = _make_client_mock(
            _mock_response(_ok(type="navidrome", serverVersion="0.61.2"))
        )
        with patch("backend.subsonic_client.httpx.Client", return_value=cm):
            client = SubsonicClient("http://navi.test", "user", "pw", "Music")
        assert client.is_connected() is True
        assert client.get_error() is None
        assert client._server_type == "navidrome"
        assert client._server_version == "0.61.2"

    def test_connect_with_invalid_credentials(self):
        cm, _ = _make_client_mock(_mock_response(_failed(40, "Wrong username or password")))
        with patch("backend.subsonic_client.httpx.Client", return_value=cm):
            client = SubsonicClient("http://navi.test", "user", "wrong", "Music")
        assert client.is_connected() is False
        assert "invalid" in client.get_error().lower()

    def test_connect_with_unreachable_server(self):
        cm = MagicMock()
        cm.__enter__ = MagicMock(side_effect=httpx.ConnectError("refused"))
        cm.__exit__ = MagicMock(return_value=False)
        with patch("backend.subsonic_client.httpx.Client", return_value=cm):
            client = SubsonicClient("http://nope.test", "user", "pw", "Music")
        assert client.is_connected() is False
        assert "cannot connect" in client.get_error().lower()

    def test_connect_with_http_error(self):
        cm, _ = _make_client_mock(_mock_response({}, status_code=500))
        with patch("backend.subsonic_client.httpx.Client", return_value=cm):
            client = SubsonicClient("http://navi.test", "user", "pw", "Music")
        assert client.is_connected() is False
        assert "500" in client.get_error()

    def test_connect_missing_url_credentials(self):
        with patch("backend.subsonic_client.httpx.Client") as mock_httpx:
            client = SubsonicClient("", "", "", "Music")
        assert client.is_connected() is False
        assert "required" in client.get_error().lower()
        mock_httpx.assert_not_called()


# ---------------------------------------------------------------------------
# Music libraries
# ---------------------------------------------------------------------------


class TestSubsonicMusicLibraries:
    """Tests for get_music_libraries via getMusicFolders.view."""

    def test_get_music_libraries_returns_folder_names(self):
        client = _make_subsonic()
        cm, _ = _make_client_mock(
            _mock_response(_ok(musicFolders={"musicFolder": [
                {"id": 1, "name": "Music Library"},
                {"id": 2, "name": "Podcasts"},
            ]}))
        )
        with patch("backend.subsonic_client.httpx.Client", return_value=cm):
            libs = client.get_music_libraries()
        assert libs == ["Music Library", "Podcasts"]

    def test_get_music_libraries_when_disconnected(self):
        client = _make_subsonic(connected=False)
        assert client.get_music_libraries() == []


# ---------------------------------------------------------------------------
# _song_to_track
# ---------------------------------------------------------------------------


class TestSubsonicSongToTrack:
    """Tests for the _song_to_track conversion."""

    def test_song_to_track_uses_displayAlbumArtist_first(self):
        client = _make_subsonic()
        track = client._song_to_track({
            "id": "abc",
            "title": "Song",
            "album": "Album",
            "duration": 200,
            "displayAlbumArtist": "Album Artist Display",
            "albumArtist": "Album Artist",
            "displayArtist": "Display Artist",
            "artist": "Artist",
        })
        assert isinstance(track, Track)
        assert track.artist == "Album Artist Display"
        assert track.duration_ms == 200_000
        assert track.art_url == "/api/art/abc"

    def test_song_to_track_falls_back_to_artist_chain(self):
        client = _make_subsonic()
        track = client._song_to_track({
            "id": "1",
            "title": "Song",
            "album": "Album",
            "duration": 100,
            "artist": "Bare Artist",
        })
        assert track.artist == "Bare Artist"

    def test_song_to_track_handles_genres_list_and_legacy_string(self):
        client = _make_subsonic()

        new_style = client._song_to_track({
            "id": "1", "title": "S", "album": "A", "duration": 0,
            "genres": [{"name": "Rock"}, {"name": "Alt"}],
            "genre": "IgnoredLegacy",
        })
        assert new_style.genres == ["Rock", "Alt"]

        legacy = client._song_to_track({
            "id": "2", "title": "S", "album": "A", "duration": 0,
            "genre": "Jazz",
        })
        assert legacy.genres == ["Jazz"]


# ---------------------------------------------------------------------------
# get_all_tracks (pagination)
# ---------------------------------------------------------------------------


class TestSubsonicGetAllTracks:
    """Tests for get_all_tracks pagination via search3.view."""

    def test_get_all_tracks_paginates_search3(self):
        client = _make_subsonic()
        page1 = [{"id": str(i), "title": f"t{i}", "album": "a", "duration": 0,
                  "artist": "x"} for i in range(PAGE_SIZE)]
        page2 = [{"id": str(i), "title": f"t{i}", "album": "a", "duration": 0,
                  "artist": "x"} for i in range(PAGE_SIZE, PAGE_SIZE + 50)]
        responses = [
            _mock_response(_ok(searchResult3={"song": page1})),
            _mock_response(_ok(searchResult3={"song": page2})),
        ]
        cm, http = _make_client_mock(responses)
        with patch("backend.subsonic_client.httpx.Client", return_value=cm):
            tracks = client.get_all_tracks()

        assert len(tracks) == PAGE_SIZE + 50
        # Verify pagination uses songOffset
        first_call_params = http.get.call_args_list[0].kwargs["params"]
        second_call_params = http.get.call_args_list[1].kwargs["params"]
        assert first_call_params["songOffset"] == 0
        assert second_call_params["songOffset"] == PAGE_SIZE

    def test_get_all_tracks_breaks_when_short_page(self):
        client = _make_subsonic()
        songs = [{"id": "1", "title": "t", "album": "a", "duration": 0, "artist": "x"}]
        cm, http = _make_client_mock([_mock_response(_ok(searchResult3={"song": songs}))])
        with patch("backend.subsonic_client.httpx.Client", return_value=cm):
            tracks = client.get_all_tracks()
        assert len(tracks) == 1
        assert http.get.call_count == 1


# ---------------------------------------------------------------------------
# get_tracks_by_filters
# ---------------------------------------------------------------------------


class TestSubsonicGetTracksByFilters:
    """Tests for get_tracks_by_filters."""

    def test_filters_by_genre_uses_getSongsByGenre(self):
        client = _make_subsonic()
        songs = [{"id": "1", "title": "Rock Song", "album": "a", "duration": 0,
                  "artist": "x", "year": 1995}]
        cm, http = _make_client_mock([_mock_response(_ok(songsByGenre={"song": songs}))])
        with patch("backend.subsonic_client.httpx.Client", return_value=cm):
            tracks = client.get_tracks_by_filters(genres=["Rock"], exclude_live=False)

        assert len(tracks) == 1
        endpoint_path = http.get.call_args_list[0].args[0]
        assert "getSongsByGenre" in endpoint_path
        assert http.get.call_args_list[0].kwargs["params"]["genre"] == "Rock"

    def test_filters_dedupe_by_id_across_genres(self):
        client = _make_subsonic()
        # Same song id "shared" appears in both genres — should only count once.
        rock_songs = [
            {"id": "shared", "title": "Crossover", "album": "a", "duration": 0, "artist": "x"},
            {"id": "rock-only", "title": "Pure Rock", "album": "a", "duration": 0, "artist": "x"},
        ]
        alt_songs = [
            {"id": "shared", "title": "Crossover", "album": "a", "duration": 0, "artist": "x"},
            {"id": "alt-only", "title": "Pure Alt", "album": "a", "duration": 0, "artist": "x"},
        ]
        cm, _ = _make_client_mock([
            _mock_response(_ok(songsByGenre={"song": rock_songs})),
            _mock_response(_ok(songsByGenre={"song": alt_songs})),
        ])
        with patch("backend.subsonic_client.httpx.Client", return_value=cm):
            tracks = client.get_tracks_by_filters(genres=["Rock", "Alt"], exclude_live=False)

        ids = {t.rating_key for t in tracks}
        assert ids == {"shared", "rock-only", "alt-only"}

    def test_filters_decade_post_filtered_in_python(self):
        client = _make_subsonic()
        songs = [
            {"id": "1", "title": "90s", "album": "a", "duration": 0, "artist": "x", "year": 1995},
            {"id": "2", "title": "80s", "album": "a", "duration": 0, "artist": "x", "year": 1985},
            {"id": "3", "title": "00s", "album": "a", "duration": 0, "artist": "x", "year": 2005},
        ]
        cm, _ = _make_client_mock([_mock_response(_ok(searchResult3={"song": songs}))])
        with patch("backend.subsonic_client.httpx.Client", return_value=cm):
            tracks = client.get_tracks_by_filters(decades=["1990s"], exclude_live=False)

        assert [t.rating_key for t in tracks] == ["1"]


# ---------------------------------------------------------------------------
# get_random_tracks
# ---------------------------------------------------------------------------


class TestSubsonicGetRandomTracks:
    """Tests for get_random_tracks via getRandomSongs.view."""

    def test_random_tracks_capped_at_500(self):
        client = _make_subsonic()
        songs = [{"id": str(i), "title": f"t{i}", "album": "a", "duration": 0,
                  "artist": "x"} for i in range(500)]
        cm, http = _make_client_mock([_mock_response(_ok(randomSongs={"song": songs}))])
        with patch("backend.subsonic_client.httpx.Client", return_value=cm):
            tracks = client.get_random_tracks(count=1000, exclude_live=False)

        assert len(tracks) == 500
        assert http.get.call_args.kwargs["params"]["size"] == 500

    def test_random_tracks_filters_live_tracks(self):
        client = _make_subsonic()
        songs = [
            {"id": "1", "title": "Live at Wembley", "album": "Live Album",
             "duration": 0, "artist": "x"},
            {"id": "2", "title": "Studio Track", "album": "Studio", "duration": 0, "artist": "x"},
            {"id": "3", "title": "Another Studio", "album": "Studio2", "duration": 0, "artist": "x"},
        ]
        cm, http = _make_client_mock([_mock_response(_ok(randomSongs={"song": songs}))])
        with patch("backend.subsonic_client.httpx.Client", return_value=cm):
            tracks = client.get_random_tracks(count=2, exclude_live=True)

        # Verify over-fetching: target = 2 * 3 = 6
        assert http.get.call_args.kwargs["params"]["size"] == 6
        ids = {t.rating_key for t in tracks}
        assert "1" not in ids  # live filtered out


# ---------------------------------------------------------------------------
# search_tracks
# ---------------------------------------------------------------------------


class TestSubsonicSearchTracks:
    """Tests for search_tracks via search3.view."""

    def test_search_tracks_returns_song_results(self):
        client = _make_subsonic()
        songs = [
            {"id": "100", "title": "Black", "album": "Ten", "duration": 340,
             "artist": "Pearl Jam", "year": 1991},
        ]
        cm, _ = _make_client_mock([_mock_response(_ok(searchResult3={"song": songs}))])
        with patch("backend.subsonic_client.httpx.Client", return_value=cm):
            results = client.search_tracks("Black")

        assert len(results) == 1
        assert results[0].title == "Black"
        assert results[0].artist == "Pearl Jam"
        assert results[0].art_url == "/api/art/100"


# ---------------------------------------------------------------------------
# count_tracks_by_filters
# ---------------------------------------------------------------------------


class TestSubsonicCountTracksByFilters:
    """Tests for count_tracks_by_filters."""

    def test_fast_path_uses_getGenres_song_count(self):
        client = _make_subsonic()
        cm, http = _make_client_mock([_mock_response(_ok(genres={"genre": [
            {"value": "Rock", "songCount": 1234},
            {"value": "Jazz", "songCount": 567},
            {"value": "Pop", "songCount": 89},
        ]}))])
        with patch("backend.subsonic_client.httpx.Client", return_value=cm):
            count = client.count_tracks_by_filters(
                genres=["Rock", "Jazz"], decades=None, exclude_live=False
            )
        assert count == 1234 + 567
        assert http.get.call_count == 1
        assert "getGenres" in http.get.call_args.args[0]

    def test_falls_back_to_full_fetch_when_decades_present(self):
        client = _make_subsonic()
        # Decade filter forces slow path → get_tracks_by_filters → search3 (no genres)
        songs = [
            {"id": "1", "title": "A", "album": "x", "duration": 0, "artist": "x", "year": 1995},
            {"id": "2", "title": "B", "album": "x", "duration": 0, "artist": "x", "year": 2005},
        ]
        cm, http = _make_client_mock([_mock_response(_ok(searchResult3={"song": songs}))])
        with patch("backend.subsonic_client.httpx.Client", return_value=cm):
            count = client.count_tracks_by_filters(
                genres=None, decades=["1990s"], exclude_live=False
            )
        assert count == 1
        assert "search3" in http.get.call_args.args[0]


# ---------------------------------------------------------------------------
# create_playlist
# ---------------------------------------------------------------------------


class TestSubsonicCreatePlaylist:
    """Tests for create_playlist."""

    def test_create_playlist_success(self):
        client = _make_subsonic()
        cm, http = _make_client_mock([
            _mock_response(_ok(playlist={"id": "pl-42", "name": "Test"})),
        ])
        with patch("backend.subsonic_client.httpx.Client", return_value=cm):
            result = client.create_playlist("Test", ["1", "2", "3"])

        assert result["success"] is True
        assert result["playlist_id"] == "pl-42"
        assert result["tracks_added"] == 3
        # Verify songId params sent (createPlaylist takes a list of tuples for repeated keys)
        sent_params = http.get.call_args_list[0].kwargs["params"]
        song_ids = [v for k, v in sent_params if k == "songId"]
        assert song_ids == ["1", "2", "3"]

    def test_create_playlist_falls_back_to_lookup_by_name(self):
        client = _make_subsonic()
        # First response: createPlaylist returns no playlist body
        # Second response: getPlaylists returns the new playlist with matching name
        cm, _ = _make_client_mock([
            _mock_response(_ok()),
            _mock_response(_ok(playlists={"playlist": [
                {"id": "pl-99", "name": "Test"},
                {"id": "pl-other", "name": "Other"},
            ]})),
        ])
        with patch("backend.subsonic_client.httpx.Client", return_value=cm):
            result = client.create_playlist("Test", ["1"])

        assert result["success"] is True
        assert result["playlist_id"] == "pl-99"


# ---------------------------------------------------------------------------
# update_playlist
# ---------------------------------------------------------------------------


class TestSubsonicUpdatePlaylist:
    """Tests for update_playlist."""

    def test_replace_mode_removes_then_adds(self):
        client = _make_subsonic()
        # 3 calls expected: getPlaylist, updatePlaylist (remove), updatePlaylist (add)
        cm, http = _make_client_mock([
            _mock_response(_ok(playlist={"id": "pl-1", "songCount": 2})),
            _mock_response(_ok()),
            _mock_response(_ok()),
        ])
        with patch("backend.subsonic_client.httpx.Client", return_value=cm):
            result = client.update_playlist("pl-1", ["new-1", "new-2"], mode="replace")

        assert result["success"] is True
        assert http.get.call_count == 3
        # Second call should have songIndexToRemove for indices 0 and 1
        remove_params = http.get.call_args_list[1].kwargs["params"]
        remove_indices = [v for k, v in remove_params if k == "songIndexToRemove"]
        assert remove_indices == ["0", "1"]
        # Third call should have songIdToAdd
        add_params = http.get.call_args_list[2].kwargs["params"]
        add_ids = [v for k, v in add_params if k == "songIdToAdd"]
        assert add_ids == ["new-1", "new-2"]

    def test_append_mode_only_adds(self):
        client = _make_subsonic()
        cm, http = _make_client_mock([_mock_response(_ok())])
        with patch("backend.subsonic_client.httpx.Client", return_value=cm):
            result = client.update_playlist("pl-1", ["new-1"], mode="append")

        assert result["success"] is True
        # Only one call (no getPlaylist, no remove)
        assert http.get.call_count == 1
        params = http.get.call_args_list[0].kwargs["params"]
        add_ids = [v for k, v in params if k == "songIdToAdd"]
        assert add_ids == ["new-1"]


# ---------------------------------------------------------------------------
# get_playlists
# ---------------------------------------------------------------------------


class TestSubsonicGetPlaylists:
    """Tests for get_playlists."""

    def test_get_playlists_sorted_by_title(self):
        client = _make_subsonic()
        cm, _ = _make_client_mock([_mock_response(_ok(playlists={"playlist": [
            {"id": "1", "name": "Zen Garden", "songCount": 5},
            {"id": "2", "name": "Afternoon Jazz", "songCount": 12},
            {"id": "3", "name": "Morning Run", "songCount": 20},
        ]}))])
        with patch("backend.subsonic_client.httpx.Client", return_value=cm):
            playlists = client.get_playlists()

        titles = [p.title for p in playlists]
        assert titles == ["Afternoon Jazz", "Morning Run", "Zen Garden"]
        for p in playlists:
            assert isinstance(p, PlexPlaylistInfo)


# ---------------------------------------------------------------------------
# get_art_url
# ---------------------------------------------------------------------------


class TestSubsonicGetArtUrl:
    """Tests for get_art_url."""

    def test_get_art_url_includes_auth_and_id(self):
        client = _make_subsonic()
        url = client.get_art_url("song-123")
        assert url is not None
        assert url.startswith("http://navi.test/rest/getCoverArt.view?")
        assert "id=song-123" in url
        assert "u=user" in url
        assert "t=" in url and "s=" in url  # token + salt baked in

    def test_get_art_url_when_disconnected(self):
        client = _make_subsonic(connected=False)
        assert client.get_art_url("song-123") is None
