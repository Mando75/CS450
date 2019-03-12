const fetch = require("node-fetch");
const { url, printProgress } = require("./utils");
const { getCourses } = require("./getCourseIds");
const { filterBaseCourses } = require("./filterBaseCourses");
const fs = require("fs").promises;

function fetchStudentEnrollments(courseIds, enrollments = [], length = null) {
  if (!length) {
    length = courseIds.length;
  }
  const progress = parseFloat(
    ((length - courseIds.length) / length) * 100
  ).toPrecision(4);
  printProgress(
    progress,
    `Fetching student enrollments: Course ${length -
      courseIds.length}/${length}`
  );
  return new Promise((resolve, reject) => {
    if (courseIds.length === 0) {
      console.log("\n");
      resolve(enrollments);
    } else {
      const courseId = courseIds.pop().id;
      fetch(url(`/courses/${courseId}/enrollments`))
        .then(res => res.json())
        .then(res => {
          return res.filter(
            ({ type, user }) =>
              type.includes("Student") && !user.name.includes("Test")
          );
        })
        .then(enrollments => {
          return enrollments.map(
            ({ user_id: userId, course_id: courseId, grades }) => ({
              userId,
              courseId,
              currentScore: grades ? grades.current_score : null,
              finalScore: grades ? grades.final_score : null,
              currentGrade: grades ? grades.current_grade : null,
              finalGrade: grades ? grades.final_grade : null
            })
          );
        })
        .then(rows => enrollments.concat(rows))
        .then(enroll =>
          fetchStudentEnrollments(courseIds, enroll, length)
            .then(resolve)
            .catch(reject)
        )
        .catch(e => {
          console.log("\n");
          return reject(e);
        });
    }
  });
}

function fetchTeacherEnrollments(courseIds, enrollments = [], length = null) {
  if (!length) {
    length = courseIds.length;
  }
  const progress = parseFloat(
    ((length - courseIds.length) / length) * 100
  ).toPrecision(4);
  printProgress(
    progress,
    `Fetching teacher enrollments: Course ${length -
      courseIds.length}/${length}`
  );

  return new Promise((resolve, reject) => {
    if (courseIds.length === 0) {
      console.log("\n");
      resolve(enrollments);
    } else {
      const courseId = courseIds.pop().id;
      fetch(
        url(`/courses/${courseId}/enrollments`, [
          {
            key: "type",
            value: "TeacherEnrollment"
          }
        ])
      )
        .then(res => res.json())
        .then(res =>
          res.map(
            ({ course_id: courseId, user_id: userId, user: { name } }) => ({
              courseId,
              userId,
              name
            })
          )
        )
        .then(row => enrollments.concat(row))
        .then(enroll => {
          fetchTeacherEnrollments(courseIds, enroll, length)
            .then(resolve)
            .catch(reject);
        });
    }
  });
}

function writeEnrollments(callback, filename) {
  return new Promise((resolve, reject) => {
    getCourses(false)
      .then(filterBaseCourses)
      .then(({ courses }) => callback(courses))
      .then(obj =>
        fs.writeFile(
          filename,
          JSON.stringify(Array.from(obj.values()), null, 2),
          "utf8"
        )
      )
      .then(resolve)
      .catch(reject);
  });
}

function getStudentEnrollments(force = false) {
  return new Promise((resolve, reject) => {
    try {
      if (force) {
        throw new Error("force refetch");
      }
      const enrollments = require("./json/studentEnrollments.json");
      resolve(enrollments);
    } catch (e) {
      console.log("Enrollments do not exist... fetching now");
      writeEnrollments(
        fetchStudentEnrollments,
        "./json/studentEnrollments.json"
      )
        .then(() => resolve(require("./json/studentEnrollments.json")))
        .catch(reject);
    }
  });
}

function getTeacherEnrollments(force = false) {
  return new Promise((resolve, reject) => {
    try {
      if (force) {
        throw new Error("force refetch");
      }
      const enrollments = require("./json/teacherEnrollments.json");
      resolve(enrollments);
    } catch (e) {
      console.log("Enrollments do not exist... fetching now");
      writeEnrollments(
        fetchTeacherEnrollments,
        "./json/teacherEnrollments.json"
      )
        .then(() => resolve(require("./json/teacherEnrollments.json")))
        .catch(reject);
    }
  });
}

module.exports = { getStudentEnrollments, getTeacherEnrollments };
