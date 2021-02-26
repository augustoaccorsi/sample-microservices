// index.js
/**
 * Required External Modules
 */
const express = require("express");
const path = require("path");
const https = require('https');
//const conn = require("./Connection.js");

/**
 * App Variables
 */
const app = express();
const port = process.env.PORT || "8000";
let value = 0;

/**
 *  App Configuration
 */
app.set("views", path.join(__dirname, "views"));
app.set("view engine", "pug");
app.use(express.static(path.join(__dirname, "public")));
/**
 * Routes Definitions
 */
app.get("/", (req, res) => {
    res.render("index", { title: "Home" });
  });

app.get("/one", (req, res) => {

  import axios from 'axios'
  let conn;

  axios.get(url).then(resp => {
    conn = resp.data;
});

    res.render("one", { title: "Microservice One", value: conn });
  });
/**
 * Server Activation
 */
app.listen(port, () => {
    console.log(`Listening to requests on http://localhost:${port}`);
  });