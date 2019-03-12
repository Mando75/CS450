require("dotenv").config();
const {
  getTeacherEnrollments,
  getStudentEnrollments
} = require("./fetchEnrollments");

// getTeacherEnrollments(true).then(enrollments =>
//   console.log("teachers ", enrollments.length)
// );
getStudentEnrollments(true).then(enrollments =>
  console.log("students", enrollments.length)
);
