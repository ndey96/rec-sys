import React, { Component } from 'react';
import './App.css';
import logo from './logo.svg';
import {authorize, getHashParams, setToken, makePlaylist} from './SpotifyFunctions.js'

class App extends Component {

  constructor() {
      super()
      this.state = {
         myText: "Generating recommendations...please wait"
      }
  }

  componentDidMount() {
   let hashParams = getHashParams();
    if(!hashParams.access_token) {
      authorize();
    } else {
      setToken(hashParams.access_token);
      makePlaylist(() => {
        this.setState({myText: 'Check Spotify for a playlist called FYDPPlaylistTest'})
      });
    }
  }

  render() {
    return ( 
      <div className="App">
        Get Rec'd
        <header className="App-header">
          {this.state.myText}
        </header>
      </div>
    );
  }
}

export default App;

