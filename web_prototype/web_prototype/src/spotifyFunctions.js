//SpotifyFunctions.js

import SpotifyWebApi from './SpotifyWebApi'

const spotifyWebApi = new SpotifyWebApi();
const stateKey = 'spotify-auth-key';
const playlistName = 'GetRecd';
const uriBuilderString = 'spotify:track:';
const devBrowser = 'http://localhost:3000/'
const prodBrowser = 'https://fydp-getrecd.appspot.com/'
const devEndpoint = 'http://0.0.0.0:8080/getrecd/'
const prodEndpoint = 'https://35.190.143.202:8080/getrecd/'
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
      getRecommendations(() => {
        callback();
      });
    });
  });
}

function getRecommendations(callback) {
  var getReq = new XMLHttpRequest();
  // Open a new connection, using the GET request on the URL endpoint
  getReq.open('GET', devEndpoint+topTrackIds.join(','), true);
  getReq.onload = function () {
    console.log(this.response)
    var res=JSON.parse(this.response)
    recommendedTrackIds = res['result']
    var uniqueRecs = Array.from(new Set(recommendedTrackIds))
    console.log(uniqueRecs)
    for (var index in uniqueRecs) {
      recommendedTrackUris.push(uriBuilderString+uniqueRecs[index])
    }
    recommendedTrackUris.shift()
    console.log(recommendedTrackUris)
    createAndSavePlaylist(() => {
       callback();
    })    
  }
  // Send request
  getReq.send();
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