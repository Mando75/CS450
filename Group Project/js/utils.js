const devToken = process.env.DEV_TOKEN;
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

function printProgress(progress, msg) {
  process.stdout.clearLine();
  process.stdout.cursorTo(0);
  process.stdout.write(msg + " " + progress + "%");
}

module.exports = {
  url,
  printProgress
};
