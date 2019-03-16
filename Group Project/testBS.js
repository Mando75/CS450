const fetch = require("node-fetch");

fetch("https://byui.brightspace.com/d2l/api/lp/1.19/courses/schema", {
  headers: {
    "X-Csrf-Token": "XlmPELnWvRtouWIvYiCUim6qFKfSl7ZN"
  }
}).then(console.log);
