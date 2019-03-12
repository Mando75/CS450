const fs = require("fs").promises;
const { fetchPaginatedData } = require("./fetchPagination");
const { url } = require("./utils");

const allCourses = account => `/accounts/${account}/courses`;
const csAccountId = 67;

/**
 * Fetches the cs courses and writes the course ids to
 * a json file
 */
function writeCourseIds() {
  return new Promise((resolve, reject) => {
    fetchPaginatedData(url(allCourses(csAccountId)))
      .then(courses =>
        courses
          .map(({ id, course_code }) => ({
            id,
            courseCode: course_code.replace(" ", "")
          }))
          .filter(
            ({ courseCode }) =>
              !courseCode.includes("Master") &&
              !courseCode.includes("2019.Spring")
          )
      )
      .then(courses =>
        fs.writeFile(
          "./json/courseIds.json",
          JSON.stringify({ courses }, null, 2),
          "utf8"
        )
      )
      .then(resolve)
      .catch(e => reject(e));
  });
}

/**
 * returns the course ids of the cs courses.
 * Checks first for an existing list of courses,
 * if they don't exist, refetches them
 */
function getCourses(force = false) {
  return new Promise((resolve, reject) => {
    try {
      if (force) {
        throw new Error("force refetch");
      }
      const ids = require("./json/courseIds.json").courses;
      resolve(ids);
    } catch (e) {
      console.log("Course Ids do not exist... fetching now");
      writeCourseIds()
        .then(() => resolve(require("./json/courseIds.json").courses))
        .catch(reject);
    }
  });
}

module.exports = {
  getCourses
};
