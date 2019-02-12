import React, { Component } from 'react';
import './App.css';
import logo from './logo.svg';
import {authorize, getHashParams, setToken, makePlaylist} from './SpotifyFunctions.js'

class App extends Component {

  constructor() {
      super()
      this.state = {
         myText: "Get Rec'd"
      }
  }

  componentDidMount() {
   let hashParams = getHashParams();
    if(!hashParams.access_token) {
      authorize();
    } else {
      console.log(hashParams.access_token)
      setToken(hashParams.access_token);
      makePlaylist();
      this.setState({myText: 'Check Spotify for a playlist called FYDPPlaylistTest'})
    }
    this.callBackendAPI()
      .then(res => this.setState({ data: res.express }))
      .catch(err => console.log(err));
  }
    // Fetches our GET route from the Express server. (Note the route we are fetching matches the GET route from server.js
  callBackendAPI = async () => {
    const response = await fetch('/');
    const body = await response.json();

    if (response.status !== 200) {
      throw Error(body.message) 
    }
    return body;
  };

  render() {
    return ( 
      <div className="App">
        {this.state.myText}
        <header className="App-header">
          <img src={logo} className="App-logo" alt="logo" />
        </header>
      </div>
    );
  }
}

export default App;

