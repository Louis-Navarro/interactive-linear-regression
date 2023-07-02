var xValues = [];
var yValues = [];

var a;
var b;

const learningRate = 0.5;
const opt = tf.train.adadelta(learningRate);

function setup() {
  createCanvas(400, 400);

  a = tf.scalar(random()).variable();
  b = tf.scalar(random()).variable();
}

function draw() {
  background(0);

  stroke(255);
  strokeWeight(8);
  for (let index = 0; index < xValues.length; index++) {
    x = map(xValues[index], 0, 1, 0, 400);
    y = map(yValues[index], 0, 1, 0, 400);
    point(x, y);
  }

  beginShape();
  strokeWeight(4);
  for (let value = 0; value <= 1.01; value += 1) {
    yPred = tf.mul(a, value).add(b);
    val = map(value, 0, 1, 0, 400);
    yPred = map(yPred.arraySync(), 0, 1, 0, 400);
    vertex(val, yPred);
  }
  endShape();

  if (xValues.length > 0) {
    optimize();
  }
}

function mousePressed() {
  x = map(mouseX, 0, 400, 0, 1);
  y = map(mouseY, 0, 400, 0, 1);

  xValues.push(x);
  yValues.push(y);
}

function forward() {
  return tf.mul(a, xValues).add(b);
}

function lossFunction() {
  preds = forward();
  return preds.sub(yValues).square().mean();
}

function optimize() {
  opt.minimize(() => lossFunction());
}
