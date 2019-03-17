const express = require('express');
const path = require('path');
const cors = require('cors');

const app = express();
app.use(cors, express.static(path.join(__dirname, 'public')));

app.get('/', function(req, res) {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.listen(9000);