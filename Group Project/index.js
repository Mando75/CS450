require("dotenv").config();
const fs = require("fs").promises;
const fetch = require("node-fetch");
const { getCourses } = require("./getCourseIds");
const { filterBaseCourses } = require("./filterBaseCourses");
const { fetchEnrollments } = require("./fetchEnrollments");
const { url } = require("./utils");

// getCourses()
//   .then(filterBaseCourses)
//   .then(({ courses }) => fetchEnrollments(courses))
//   .then(obj =>
//     fs.writeFile(
//       "./json/enrollments.json",
//       JSON.stringify(obj, null, 2),
//       "utf8"
//     )
//   );
fetch(
  url(`/courses/${33302}/enrollments`, [
    { key: "type", value: "TeacherEnrollment" }
  ])
)
  .then(res => res.json())
  .then(res =>
    fs.writeFile("./json/test.json", JSON.stringify(res, null, 2), "utf8")
  );

// const enrollments = require("./json/enrollments.json");
// console.log(enrollments.filter(e => !e.teacherId).length);
// console.log(enrollments.filter(e => e.teacherId).length);
