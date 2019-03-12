//SpotifyFunctions.js

import SpotifyWebApi from './SpotifyWebApi'

const spotifyWebApi = new SpotifyWebApi();
const stateKey = 'spotify-auth-key';
const playlistName = 'FYDPPlaylistTest';
const uriBuilderString = 'spotify:track:';
const devBrowser = 'http://localhost:3000/'
const prodBrowser = 'https://fydp-getrecd.appspot.com/'
var topTrackIds = [String];
var recommendedTrackIds = [String];
var recommendedTrackUris = [String];

export function authorize(callback) {
      localStorage.removeItem(stateKey);
      const CLIENT_ID = '3f2b320cfc4f4af5a106fa21e6bc8d0c';
      const REDIRECT_URI = devBrowser;
      const scopes = [
      'user-top-read',
      'playlist-modify-public',
      'playlist-modify-private'];

      const generateRandomString = length => {
          let text = '';
          const possible =
          'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';

      const stateKey = 'spotify-auth-key'
      var access_token = ''


          while (text.length <= length) {
           text += possible.charAt(Math.floor(Math.random() * possible.length));
          }

          return text;
        };
      const state = generateRandomString(16);
      localStorage.setItem(stateKey, state);

      const url = 'https://accounts.spotify.com/authorize?client_id=' + CLIENT_ID +
        '&redirect_uri=' + encodeURIComponent(REDIRECT_URI) +
        '&scope=' + encodeURIComponent(scopes.join(' ')) +
        '&response_type=token' + '&state='+encodeURIComponent(state);

      window.location.href = url;
  }

export function getHashParams() {
    var hashParams = {};
    var e, r = /([^&;=]+)=?([^&;]*)/g,
    q = window.location.hash.substring(1);
    while ( e = r.exec(q)) {
      hashParams[e[1]] = decodeURIComponent(e[2]);
      }
    return hashParams;
  }


export function setToken(token: String) {
  spotifyWebApi.setAccessToken(token);
}

export function makePlaylist(callback) {
  getTopTracks(() => {
    callback()
  });
}

function getTopTracks(callback) {
  spotifyWebApi.getMyTopTracks({"limit": 49})
    .then((response) => {
      for (var index in response['items']) {
        topTrackIds.push(response['items'][index]['id'])
      }
      spotifyWebApi.getMyTopTracks({"limit": 50,
                                    "offset": 49})
      .then((response) => {
      for (var index in response['items']) {
        topTrackIds.push(response['items'][index]['id'])
      }
      topTrackIds.shift()
      console.log(topTrackIds)
      getRecommendations(() => {
        callback();
      });
    });
  });
}

// TODO: Call python script here
function getRecommendations(callback) {
  recommendedTrackIds = topTrackIds
  for (var index in recommendedTrackIds) {
    recommendedTrackUris.push(uriBuilderString+recommendedTrackIds[index])
  }
  recommendedTrackUris.shift()
  console.log(recommendedTrackUris)
  createAndSavePlaylist(() => {
      callback();
  });
}

function createAndSavePlaylist(callback) {
  spotifyWebApi.createPlaylist({"name": playlistName})
  .then((response) => {
    console.log(response)
    spotifyWebApi.addTracksToPlaylist(response['id'], {"uris": recommendedTrackUris})
    .then((response) => {
      console.log(response)
      callback();
    });
  });
}