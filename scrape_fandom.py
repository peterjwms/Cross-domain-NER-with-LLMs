# used to scrape various sites hosted on fandom for use in NER data for our project
# probably star wars, star trek, and some smaller site


import asyncio
import os
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from requests_html import AsyncHTMLSession


async def retrieve_start_page_links(link, type="category"):
    # browser = await launch()
    session = AsyncHTMLSession()
    # page = requests.get(url=link)
    response = await session.get(link)
    await response.html.arender(sleep=2, timeout=20)
    page = response.html.html
    await session.close()
    # print(page)

    # with open(Path("raw.html"), "w", encoding='utf-8') as f: f.write(page)
    
    soup = BeautifulSoup(page, "html.parser")
    if type == "intersection":
        # with open(Path("temp.txt"), "w") as f:
        #     print("writing") 
        #     f.write(soup.text)
        link_holder = soup.find(class_="mw-category mw-category-columns")
    elif type == "category":
        link_holder = soup.find(class_="category-page__members")
    links = {a['href'] for a in link_holder.find_all("a", href=True)}
    
    return links


def retrieve_page_content(link, filename, folder):

    # session = AsyncHTMLSession()
    page = requests.get(url=link)
    # response = await session.get(link)
    # await response.html.arender(sleep=1, timeout=20)
    # page = response.html.raw_html
    # await session.close()
    soup = BeautifulSoup(page.text, "html.parser", from_encoding=page.encoding)

    content = soup.find(class_="mw-content-ltr mw-parser-output")
    # print(content.text)
    # from Original Trilogy page
    # want text from <p>, <ul>, <blockquote>
    # question - include <div class="quote"> sections? look like they're all real-life quotes that probably don't have much to annotate
    # should stop at <h2> w/ a "Sources" span
    
    if not os.path.exists(Path(folder)):
        os.makedirs(folder)

    with open(Path(f"{folder}/{filename}.txt"), "w", encoding=page.encoding) as f:
        # make sure the file exists as empty
        pass

    with open(Path(f"{folder}/{filename}.txt"), "a", encoding=page.encoding) as f:
        stop_headers = ["Sources[]", "Credits[]", "Appearances[]", "Notes and references[]", "External links[]",
                        "Sources", "Credits", "Appearances", "Notes and references", "External links",
                        "References", "References[]", "Spacecraft references", "Spacecraft references[]",
                        "Unreferenced material", "Unreferenced material[]"]
        # for child in content.find_all(["p", "ul", "blockquote", "h2", "h3", "h4"], recursive=False):
        for child in content.children:
            if child.text.strip() == "": 
                continue
            if child.name == "aside" or child.name is None: 
                continue
            elif child.name not in ["p", "blockquote"]:
                continue
            if child.text.strip() in stop_headers:
                print(child.name, child.text)
                # don't scrape any non-paragraph info, don't need sources, cedits, notes, references, etc.
                break
            # f.write(f"{child.name}\n")
            f.write(f"{child.text}\n")

    return

    

async def main():
    # TODO: these are the links/presets to change to scrape another site
    sw_stem = "https://starwars.fandom.com"
    sw_films = "/wiki/Category:Saga_films"
    sw_best = "/wiki/Special:BlankPage/CategoryIntersection?category1=Wookieepedia_Featured_articles&category2=Canon_articles&category3="
    sw_featured = "/wiki/Category:Wookieepedia_Featured_articles"
    
    st_stem = "https://memory-alpha.fandom.com"
    st_films = "/wiki/Category:Star_Trek_films"
    

    site_stem = st_stem
    start_page = st_films
    start_page = f"{site_stem}{start_page}"
    folder = "star_trek_films"
    
    links = await retrieve_start_page_links(start_page, "category")

    for i, link in enumerate(links):
        print(f"link is {site_stem}{link}")
        # filename removes the "/wiki/" at the front of the link
        filename = str.replace(link[6:], ":", "_").replace("/", "_")
        print(f"filename is {filename}\n")


        retrieve_page_content(f"{site_stem}{link}", filename, folder)
        # if i == 0: break



if __name__ == "__main__":
    asyncio.run(main())
    # main()