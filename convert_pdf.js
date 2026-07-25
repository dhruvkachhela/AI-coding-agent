const puppeteer = require('puppeteer');
const path = require('path');

(async () => {
    try {
        console.log("Launching headless browser...");
        const browser = await puppeteer.launch({
            headless: 'new',
            args: ['--no-sandbox', '--disable-setuid-sandbox']
        });
        const page = await browser.newPage();
        
        // Load the HTML file
        const filePath = path.resolve('pdf_source.html');
        console.log(`Loading HTML from: file://${filePath}`);
        await page.goto(`file://${filePath}`, { waitUntil: 'networkidle0' });
        
        // Give time for images to load cleanly
        await new Promise(r => setTimeout(r, 2000));
        
        // Generate PDF
        await page.pdf({
            path: 'documentation.pdf',
            format: 'A4',
            printBackground: true,
            margin: { top: '15mm', right: '12mm', bottom: '15mm', left: '12mm' }
        });
        
        console.log("PDF created successfully at documentation.pdf!");
        await browser.close();
    } catch (e) {
        console.error("Error creating PDF:", e);
        process.exit(1);
    }
})();
