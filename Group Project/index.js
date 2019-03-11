require("dotenv").config();
const fetch = require("node-fetch");
// const { getCourses } = require("./getCourseIds");
// const { filterBaseCourses } = require("./filterBaseCourses");

// getCourses()
// .then(filterBaseCourses)
// .then(() => {
//
// });
const url = (path, params = []) => {
  const devToken = process.env.DEV_TOKEN;
  let url = `https://byui.instructure.com/api/v1${path}?access_token=${devToken}&per_page=50&exclude_blueprint_courses=true`;
  if (params) {
    params.forEach(param => {
      url += `&${param.key}=${param.value}`;
    });
  }
  return url;
};

fetch(url("/courses/11157/enrollments"))
  .then(res => res.json())
  .then(res =>
    res.filter(
      ({ type, user }) =>
        type.includes("Student") && !user.name.includes("Test")
    )
  )
  .then(enrollments =>
  enrollments.map(({ user_id: userId, course_id: courseId, grades: {current_score: currentScore, final_score: finalScore} }) => ({
  userId,
  courseId,
  currentScore,
  finalScore
  }))
  )
  .then(filtered => console.log(filtered));
