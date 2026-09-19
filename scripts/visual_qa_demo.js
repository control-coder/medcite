const { chromium } = require("playwright");
const path = require("path");
const fs = require("fs");

const baseUrl = process.env.MEDIDIAG_DEMO_URL || "http://127.0.0.1:8400";
// 截图按职责写入产物目录，避免默认污染项目根目录。
const outputDir = process.env.MEDIDIAG_VISUAL_OUTPUT || path.resolve(__dirname, "../artifacts/visual");
const executablePath = process.env.MEDIDIAG_CHROME_EXECUTABLE;
const existingCaseUrl = process.env.MEDIDIAG_CASE_URL;
const scenario = process.env.MEDIDIAG_VISUAL_SCENARIO || "active";
const expectedPolling = process.env.MEDIDIAG_EXPECT_POLLING || "every 2s";
const disableJavaScript = process.env.MEDIDIAG_DISABLE_JS === "true";

async function layoutMetrics(page) {
  return page.evaluate(() => ({
    viewportWidth: document.documentElement.clientWidth,
    scrollWidth: document.documentElement.scrollWidth,
    bodyWidth: document.body.getBoundingClientRect().width,
    htmxLoaded: typeof window.htmx !== "undefined",
  }));
}

(async () => {
  fs.mkdirSync(outputDir, { recursive: true });
  const browser = await chromium.launch({
    headless: true,
    ...(executablePath ? { executablePath } : {}),
  });
  const consoleErrors = [];
  const requests = [];
  let validationErrorVerified = false;
  try {
    const desktop = await browser.newPage({
      viewport: { width: 1440, height: 900 },
      javaScriptEnabled: !disableJavaScript,
    });
    desktop.on("console", (message) => {
      if (message.type() === "error") consoleErrors.push(message.text());
    });
    desktop.on("request", (request) => {
      if (request.url().startsWith(baseUrl)) {
        requests.push(`${request.method()} ${request.url()}`);
      }
    });
    await desktop.goto(`${baseUrl}/demo`, { waitUntil: "networkidle" });
    await desktop.screenshot({
      path: path.join(outputDir, "medidiag-demo-home-desktop.png"),
      fullPage: true,
    });
    const home = await layoutMetrics(desktop);

    let caseUrl = existingCaseUrl;
    if (!caseUrl) {
      await desktop.fill("#question", "Deidentified simulated case for browser layout verification.");
      if (!disableJavaScript) {
        await desktop.selectOption("#input-kind", "public_dataset");
        const errorResponsePromise = desktop.waitForResponse(
          (response) =>
            response.request().method() === "POST" &&
            response.url().endsWith("/demo/cases") &&
            response.status() === 400,
          { timeout: 5000 },
        );
        await desktop.click('button[type="submit"]');
        await errorResponsePromise;
        await desktop.locator("#form-error .error-banner").waitFor({ timeout: 5000 });
        validationErrorVerified = true;
      }
      await desktop.selectOption("#input-kind", "deidentified_simulation");
      const formValidity = await desktop.locator(".case-form").evaluate((form) => form.checkValidity());
      if (!formValidity) {
        const invalidFields = await desktop.locator(".case-form :invalid").evaluateAll((items) =>
          items.map((item) => ({ name: item.name, message: item.validationMessage, value: item.value })),
        );
        throw new Error(JSON.stringify({ message: "demo form is invalid", invalidFields }));
      }
      const responsePromise = desktop.waitForResponse(
        (response) =>
          response.request().method() === "POST" &&
          response.url().endsWith("/demo/cases"),
        { timeout: 5000 },
      );
      await desktop.click('button[type="submit"]');
      let submitResponse;
      try {
        submitResponse = await responsePromise;
      } catch (error) {
        throw new Error(
          JSON.stringify({
            message: "demo form emitted no POST response",
            url: desktop.url(),
            htmxLoaded: await desktop.evaluate(() => typeof window.htmx !== "undefined"),
            requests,
            consoleErrors,
          }),
        );
      }
      await desktop.waitForTimeout(1000);
      if (!/\/demo\/cases\//.test(desktop.url())) {
        const body = await submitResponse.text();
        throw new Error(
          JSON.stringify({
            message: "demo form did not redirect",
            status: submitResponse.status(),
            body,
            formValidity,
            htmxLoaded: await desktop.evaluate(() => typeof window.htmx !== "undefined"),
            formError: await desktop.locator("#form-error").innerText(),
          }),
        );
      }
      caseUrl = desktop.url();
    } else {
      await desktop.goto(caseUrl, { waitUntil: "networkidle" });
    }
    await desktop.screenshot({
      path: path.join(outputDir, `medidiag-demo-${scenario}-desktop.png`),
      fullPage: true,
    });
    const activeDesktop = await layoutMetrics(desktop);
    const polling = await desktop.locator("#case-live").getAttribute("hx-trigger");
    const visible = {
      supported: (await desktop.getByText("SUPPORTED", { exact: true }).count()) > 0,
      finalReport: (await desktop.getByText("最终结构化报告", { exact: true }).count()) > 0,
      disclaimer: (await desktop.getByText("仅供学习和工程演示，不构成医疗建议。", { exact: true }).count()) > 0,
      humanForm: (await desktop.getByText("人工处置", { exact: true }).count()) > 0,
    };

    const mobile = await browser.newPage({
      viewport: { width: 390, height: 844 },
      javaScriptEnabled: !disableJavaScript,
    });
    mobile.on("console", (message) => {
      if (message.type() === "error") consoleErrors.push(message.text());
    });
    await mobile.goto(caseUrl, { waitUntil: "networkidle" });
    await mobile.screenshot({
      path: path.join(outputDir, `medidiag-demo-${scenario}-mobile.png`),
      fullPage: true,
    });
    const activeMobile = await layoutMetrics(mobile);

    const unexpectedConsoleErrors = consoleErrors.filter(
      (message) =>
        !(validationErrorVerified && message.includes("400 (Bad Request)")),
    );
    const result = {
      caseUrl,
      home,
      activeDesktop,
      activeMobile,
      polling,
      visible,
      disableJavaScript,
      validationErrorVerified,
      expectedClientErrors: validationErrorVerified
        ? consoleErrors.filter((message) => message.includes("400 (Bad Request)"))
        : [],
      consoleErrors: unexpectedConsoleErrors,
    };
    process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
    if (
      home.scrollWidth > home.viewportWidth ||
      activeDesktop.scrollWidth > activeDesktop.viewportWidth ||
      activeMobile.scrollWidth > activeMobile.viewportWidth ||
      (expectedPolling === "none" ? polling !== null : polling !== expectedPolling) ||
      unexpectedConsoleErrors.length > 0
    ) {
      process.exitCode = 1;
    }
  } finally {
    await browser.close();
  }
})();
