const request = require("supertest");
const { expect } = require("chai");
const { Worker } = require("worker_threads");
const app = require("../../server");

describe("Anonymize worker lifecycle", function () {
  this.timeout(15000);

  let originalPostMessage;
  let exits;

  // the controller creates one worker per request and posts it one message,
  // so hooking postMessage lets us watch each worker it starts
  beforeEach(function () {
    exits = [];
    originalPostMessage = Worker.prototype.postMessage;
    Worker.prototype.postMessage = function (...args) {
      exits.push(new Promise((resolve) => this.once("exit", resolve)));
      return originalPostMessage.apply(this, args);
    };
  });

  afterEach(function () {
    Worker.prototype.postMessage = originalPostMessage;
  });

  const exitCodes = () =>
    Promise.all(
      exits.map((exited) =>
        Promise.race([exited, new Promise((r) => setTimeout(() => r("still running"), 5000))])
      )
    );

  it("should let the worker exit after a successful anonymization", async function () {
    const csv = Buffer.from("Patient ID,Notes\nP1,ok\n");
    const res = await request(app).post("/api/anonymize").attach("file", csv, "ok.csv");

    expect(res.status).to.equal(200);
    expect(await exitCodes()).to.deep.equal([0]);
  });

  it("should let the worker exit when the file cannot be read", async function () {
    const corrupt = Buffer.concat([Buffer.from([0x50, 0x4b, 0x03, 0x04]), Buffer.alloc(200, 7)]);
    const res = await request(app).post("/api/anonymize").attach("file", corrupt, "bad.xlsx");

    expect(res.status).to.equal(400);
    expect(await exitCodes()).to.deep.equal([0]);
  });
});
