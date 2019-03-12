const fetch = require("node-fetch");
const { url, printProgress } = require("./utils");

function fetchStudentEnrollments(courseIds, enrollments = [], length = null) {
  if (!length) {
    length = courseIds.length;
  }
  const progress = parseFloat(
    ((length - courseIds.length) / length) * 100
  ).toPrecision(4);
  printProgress(
    progress,
    `Fetching enrollments: Course ${length - courseIds.length}/${length}`
  );
  return new Promise((resolve, reject) => {
    if (courseIds.length === 0) {
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
              finalScore: grades ? grades.final_score : null
            })
          );
        })
        .then(rows => enrollments.concat(rows))
        .then(enroll =>
          fetchEnrollments(courseIds, enroll, length)
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
    length = courseIds.length
  }
  return new Promise((resolve, reject) => {

  })
}

module.exports = { fetchStudentEnrollments };
