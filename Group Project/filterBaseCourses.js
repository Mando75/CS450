const fs = require("fs").promises;

function filterBaseCourses(courses) {
  return new Promise((resolve, reject) => {
    console.log(courses.length);
    const cs124 = courses.filter(({ courseCode }) =>
      courseCode.includes("124")
    );
    const cs165 = courses.filter(({ courseCode }) =>
      courseCode.includes("165")
    );
    const cs235 = courses.filter(({ courseCode }) =>
      courseCode.includes("235")
    );
    fs.writeFile("./json/cs124.json", JSON.stringify(cs124, null, 2), "utf8")
      .then(() =>
        fs.writeFile(
          "./json/cs165.json",
          JSON.stringify(cs165, null, 2),
          "utf8"
        )
      )
      .then(() =>
        fs.writeFile(
          "./json/cs235.json",
          JSON.stringify(cs235, null, 2),
          "utf8"
        )
      )
      .then(resolve)
      .catch(reject);
  });
}

module.exports = { filterBaseCourses };
