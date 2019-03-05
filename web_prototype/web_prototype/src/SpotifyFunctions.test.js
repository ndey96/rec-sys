import SpotifyWebApi from './SpotifyWebApi'

function addToPlaylist() {
    spotifyWebApi.addTracksToPlaylist('7u75nsMS1xKqKMkvqCa05h' {"uris": ['5zaIgI9HNUPIcfaeVRlxGa']})
    .then((response) => {
      console.log(response)
    });
}

function createPlaylist() {
	spotifyWebApi.createPlaylist({"name": "test"})
  	.then((response) => {
  	console.log(response)
  });
}

function getTopTracks() {
  spotifyWebApi.getMyTopTracks()
    .then((response) => {
      console.log(response)
    });
}