import React, { Component} from 'react';
import './App.css';
import logo from './logo.svg';
import {authorize, getHashParams, setToken, makePlaylist} from './SpotifyFunctions.js'

class App extends Component {

  constructor() {
      super()
      this.state = {
         myText: 'Select the button below to generate recommendations. You will be redirected to authorize via Spotify.',
         shown: 'block'
      }
      console.log(this.state)
  }

  componentDidMount() {
   let hashParams = getHashParams();
    if(hashParams.access_token) {
      this.setState({})
      console.log(this.state)
      this.setState({myText: 'Generating recommendations...please wait',
                     shown: 'none'})
      setToken(hashParams.access_token);
      console.log(this.state)
      makePlaylist(() => {
        this.setState({myText: 'Check Spotify in a few minutes for a playlist called FYDPPlaylistTest'})
      });
    }
  }

  render() {
    const style = {display: this.state.shown}
    return ( 
      <div className="App">
        Get Rec'd
        <header className="App-header">
          {this.state.myText}
           <button style={style} className="Button-header" onClick={authorize}>
              Get Rec'd
            </button>
        </header>
       
      </div>
    );
  }
}

export default App;

