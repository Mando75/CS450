require("dotenv").config();
const { getCourseIds } = require("./getCourseIds");

getCourseIds().then(courseIds => {
  console.log(courseIds.length);
});
