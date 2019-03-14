require("dotenv").config();
const courseIds = require("./json/courseIds.json").courses;
const studentEnrollment = require("./json/studentEnrollments.json");
const teacherEnrollment = require("./json/teacherEnrollments.json");
const fs = require("fs").promises;

function appendAvgGrades() {
  courseIds.forEach(({ id }) => {
    const courses = studentEnrollment.filter(({ courseId }) => courseId === id);
    const avgGrade = getAvgGrade(courses);
    studentEnrollment.forEach(stu => {
      if (stu.courseId === id) {
        stu.courseAvgGrade = avgGrade;
      }
    });
  });
}

function getAvgGrade(courses) {
  const sum = courses.reduce((prev, curr) => {
    const score = curr.finalScore || curr.currentScore;
    return prev + (score ? score : 0);
  }, 0);
  return sum / courses.length;
}

function appendTeacherId() {
  studentEnrollment.forEach(stu => {
    const teach = teacherEnrollment.find(te => te.courseId === stu.courseId);
    stu.teacherId = teach ? teach.userId : null;
    stu.teacherName = teach ? teach.name : null;
  });
}

function appendCourseName() {
  studentEnrollment.forEach(stu => {
    const course = courseIds.find(c => c.id === stu.courseId);
    stu.courseCode = course ? course.courseCode : null;
  });
}

appendAvgGrades();
appendTeacherId();
appendCourseName();

fs.writeFile(
  "./json/linkedEnrollments.json",
  JSON.stringify(studentEnrollment, null, 2),
  "utf8"
).then(() => {
  console.log("Success");
  console.log(require("./json/linkedEnrollments.json").length);
});
