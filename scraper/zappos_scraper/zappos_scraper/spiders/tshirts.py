# scraper/zappos_scraper/spiders/tshirts.py
import scrapy
import json
from scrapy_playwright.page import PageMethod

class TshirtsSpider(scrapy.Spider):
    name = 'tshirts'
    allowed_domains = ['zappos.com']
    
    # We no longer use start_urls, we use start_requests to pass Playwright metadata
    def start_requests(self):
        url = 'https://www.zappos.com/men-shirts-tops/CKvXARDL1wHAAQLiAgMBAhg.zso?s=isNew%2Fdesc%2FgoLiveDate%2Fdesc%2FrecentSalesStyle%2Fdesc%2F'
        yield scrapy.Request(
            url,
            meta=dict(
                playwright=True,
                playwright_include_page=True,
                playwright_page_methods=[
                    # This tells Playwright to wait until the network is idle,
                    # which is a good sign that JS loading is complete.
                    PageMethod('wait_for_load_state', 'networkidle')
                ],
                errback=self.errback,
            )
        )

    async def parse(self, response):
        # We need to close the page context that Playwright opens
        page = response.meta["playwright_page"]
        await page.close()

        # The selectors from the previous attempt should now work on the fully-rendered HTML
        product_links = response.css('article a.pl-a::attr(href)').getall()
        self.logger.info(f"Found {len(product_links)} product links on page {response.url}")
        
        for link in product_links:
            # Product pages might not need Playwright, so we can use a standard request
            yield response.follow(link, self.parse_product)

        next_page = response.css('a[rel="next"]::attr(href)').get()
        if next_page is not None:
            yield response.follow(next_page, self.parse, meta=dict(
                playwright=True,
                playwright_include_page=True,
                playwright_page_methods=[
                    PageMethod('wait_for_load_state', 'networkidle')
                ],
                errback=self.errback,
            ))

    # This is a robust error handler for Playwright
    async def errback(self, failure):
        page = failure.request.meta["playwright_page"]
        await page.close()
        self.logger.error(f"Playwright request failed: {failure.request.url}")


    def parse_product(self, response):
        # This function can remain the same as the previous version, as product pages
        # are often simpler and might not require JS rendering.
        attributes = {}
        for li in response.css('div#productDetails ul li'):
            text = " ".join(li.css('*::text').getall())
            if text and ':' in text:
                parts = text.split(':', 1)
                key = parts[0].strip().lower().replace(' ', '_')
                value = parts[1].strip().replace('.', '')
                attributes[key] = value

        image_url = response.css('img[data-zapp-id="main-product-image"]::attr(src)').get()
        title = response.css('h1[data-zapp-id="product-title"]::text').get()
        description_parts = response.css('div#productDescription ::text').getall()
        description = "".join(description_parts).strip()

        if attributes and image_url and title:
            yield {
                'title': title,
                'description': description,
                'image_url': image_url,
                'attributes_json': json.dumps(attributes)
            }