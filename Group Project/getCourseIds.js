const fs = require("fs").promises;
const { fetchPaginatedData } = require("./fetchPagination");

const devToken = process.env.DEV_TOKEN;

const url = (path, params = []) => {
  let url = `https://byui.instructure.com/api/v1${path}?access_token=${devToken}&per_page=50&exclude_blueprint_courses=true`;
  if (params) {
    params.forEach(param => {
      url += `&${param.key}=${param.value}`;
    });
  }
  return url;
};

const allCourses = account => `/accounts/${account}/courses`;
const csAccountId = 67;

/**
 * Fetches the cs courses and writes the course ids to
 * a json file
 */
function writeCourseIds() {
  return new Promise((resolve, reject) => {
    fetchPaginatedData(url(allCourses(csAccountId)))
      .then(courses => courses.map(course => course.id))
      .then(courses =>
        fs.writeFile(
          "./json/courseIds.json",
          JSON.stringify({ courses }),
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
function getCourseIds() {
  return new Promise((resolve, reject) => {
    try {
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
  getCourseIds
};
