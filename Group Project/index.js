const parseLink = require("parse-link-header");
const fetch = require("node-fetch");
require("dotenv").config();

const devToken = process.env.DEV_TOKEN;
const url = (path, params = []) => {
  let url = `https://byui.instructure.com/api/v1${path}?access_token=${devToken}`;
  if (params) {
    params.forEach(param => {
      url += `&${param.key}=${param.value}`;
    });
  }
  return url;
};

const allCourses = account => `/accounts/${account}/courses`;
const csAccountId = 67;

function fetchPaginatedData(url, data = []) {
  return new Promise((resolve, reject) =>
    fetch(url).then(res => {
      // Reject when fetch error
      if (res.status !== 200) {
        console.log(res.status, res.statusText);
        reject(new Error("Bad fetch"));
      }
      // Get pagination link
      const link = parseLink(res.headers.get("Link"));
      res.json().then(page => {
        // Add response data to list
        data = data.concat(page);
        // Determine progress
        const progress = parseFloat(
          (parseInt(link.current.page) / parseInt(link.last.page)) * 100
        ).toPrecision(4);
        printProgress(
          progress,
          `Progress: Page ${link.current.page}/${link.last.page}`
        );
        // If we are not on the last page, recursively call
        // fetch on next link
        if (link.current.page !== link.last.page) {
          // Recursive call. On resolution/rejected, pass it
          // up the promise resolution/rejection chain
          fetchPaginatedData(link.next.url + `&access_token=${devToken}`, data)
            .then(resolve)
            .catch(reject);
        } else {
          // Otherwise, we're done. Reset stdout
          // resolve
          console.log("\n");
          resolve(data);
        }
      });
    })
  );
}

function printProgress(progress, msg) {
  process.stdout.clearLine();
  process.stdout.cursorTo(0);
  process.stdout.write(msg + " " + progress + "%");
}

fetchPaginatedData(url(allCourses(csAccountId)))
  .then(c => console.log("done", c.length))
  .catch(e => console.log(e));
