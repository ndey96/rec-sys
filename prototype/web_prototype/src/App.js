import React, { Component } from 'react';
import './App.css';
import {authorize} from './spotifyFunctions.js'
import { setToken } from './tokenActions';

class App extends Component {

  componentDidMount() {
    let hashParams = {};
    let e, r = /([^&;=]+)=?([^&;]*)/g,
      q = window.location.hash.substring(1);
    while ( e = r.exec(q)) {
      hashParams[e[1]] = decodeURIComponent(e[2]);
    }

    if(!hashParams.access_token) {
      authorize();
    } else {
      console.log(hashParams.access_token)
      //run playlist generation
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
      Get Rec'd
      </div>
    );
  }
}

export default App;

