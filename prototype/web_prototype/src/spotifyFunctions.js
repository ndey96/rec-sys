//spotifyFunctions.js

const stateKey = 'spotify-auth-key'
var access_token = ''

export function authorize() {
      localStorage.removeItem(stateKey);
      const CLIENT_ID = "3f2b320cfc4f4af5a106fa21e6bc8d0c";
      const REDIRECT_URI = "http://localhost:3000/";
      const scopes = [
      "user-top-read",
      "user-modify-private"];

      const generateRandomString = length => {
          let text = "";
          const possible =
          "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

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

  function getHashParams() {
    var hashParams = {};
    var e, r = /([^&;=]+)=?([^&;]*)/g,
    q = window.location.hash.substring(1);
    while ( e = r.exec(q)) {
      hashParams[e[1]] = decodeURIComponent(e[2]);
      }
    return hashParams;
  }

  export function getAccessToken() {
    var params = getHashParams()
    access_token = params.access_token; 
    return access_token
  }