import sys, os, time, requests, markdownify, re, json
from langchain.docstore.document import Document
from dotenv import load_dotenv
load_dotenv()

class NHSMedicationAPI:
    def __init__(self):
        load_dotenv()
        self.api_key = self._load_api_key()
        self.base_url = "https://api.nhs.uk/medicines"
        self.base_params = {
            "subscription-key": self.api_key,
        }
        self.date_last_run = os.getenv("DATE_LAST_RUN")

    def _load_api_key(self):
        if "NHS_API_KEY" in os.environ:
            print("NHS API Key Loaded")
            return os.environ["NHS_API_KEY"]
        else:
            print("NHS API Key Missing from .env")
            sys.exit()

    def get_medication_list(self):
        medication_table = {"data": []}
        prev_page_medication = ""
        
        for page in range(1, 100):
            try:
                self.base_params["page"] = page
                response = requests.get(self.base_url, params=self.base_params)
                response.raise_for_status()
                results = response.json()
                
                if prev_page_medication == results["significantLink"][-1]["name"]:
                    break
                
                for object in results["significantLink"]:
                    name, url, dateModified = object["name"], object["url"], object["mainEntityOfPage"]["dateModified"]
                    medication_table["data"].append({"name": name, "url": url, "dateModified": dateModified})
                    print(f"Drug: {name}, URL: {url}, Date Modified: {dateModified}")
                    prev_page_medication = name
            except requests.exceptions.RequestException as e:
                print(f"Error occurred while fetching medication list: {e}")
            except ValueError as e:
                print(f"Error occurred while parsing JSON response for medication list: {e}")
            except Exception as e:
                print(f"An unexpected error occurred for API request to medication list: {e}")
            
            time.sleep(7)
        
        self._save_medication_table(medication_table)

    def _save_medication_table(self, medication_table):
        with open('testdata/NHSmed/medication_table.json', 'w', encoding='utf-8') as json_file:
            json.dump(medication_table, json_file, ensure_ascii=False, indent=4)

    def load_med_list(self):
        with open('testdata/NHSmed/medication_table.json', 'r', encoding='utf-8') as json_file:
            return json.load(json_file)

    def get_all_medications(self, medication_table):
        for med in medication_table["data"]:
            self._process_medication(med)

    def _process_medication(self, med):
        url = med["url"]
        try:
            response = requests.get(url, params=self.base_params)
            response.raise_for_status()
            results = response.json()
            
            name, description, url = results['name'], results['description'], results['url']
            alternateName = " ".join(results['about']['alternateName'])
            
            
            whole_page = self._create_page_header(name, description, alternateName)
            documentjson = {}
            
            for section in results['hasPart']:
                whole_page, documentjson = self._process_section(section, name, description, alternateName, whole_page, documentjson)
            
            self._save_markdown(name, whole_page)
            self._save_json(name, documentjson)

        except requests.exceptions.RequestException as e:
            print(f"Error occurred while fetching data for {med}: {e}")
        except ValueError as e:
            print(f"Error occurred while parsing JSON response for {med}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred for {med}: {e}")
        
        time.sleep(7)

    def _create_page_header(self, name, description, alternateName):
        if alternateName == "":
            return f"# {name}\n\n## {description}\n\n"
        else:
            return f"# {name} ({alternateName})\n\n## {description}\n\n"

    def _process_section(self, section, name, description, alternateName, whole_page, documentjson):
        subdescription = section.get("description", "")
        headline = section.get("headline", "")
        apiurl = section.get("url", "")
        suburl = re.sub(r'/api.', r'/', apiurl)
        
        paragraph_content = self._create_paragraph_header(headline, subdescription)
        
        for paragraph in section["hasPart"]:
            paragraph_content += self._process_paragraph(paragraph)
        
        whole_page += paragraph_content

        titlefromurl = self._get_title_from_url(suburl, headline)
        
        doc = self._create_document(paragraph_content, name, suburl, alternateName, description, titlefromurl)
        documentjson[titlefromurl] = doc.json()
        print(f"Document created for {name} - {titlefromurl}")
        
        return whole_page, documentjson

    def _create_paragraph_header(self, headline, subdescription):
        if headline == "":
            return f"**{subdescription}**\n\n"
        else:
            return f"## {headline}\n\n**{subdescription}**\n\n"

    def _process_paragraph(self, paragraph):
        if paragraph.get("@type") == "Question":
            text = paragraph["acceptedAnswer"].get("text", "")
            subhead = paragraph.get("name", "")
        else:
            subhead = paragraph.get("headline","")
            text = paragraph.get("text", "")
        
        mdtext = markdownify.markdownify(text)
        cleaned_md = re.sub(r'\[(.*?)\](.*?\))', r'\1', mdtext)
        
        if subhead == "":
            return f"{cleaned_md}\n\n"
        else:
            return f"### {subhead}\n\n{cleaned_md}\n\n"

    def _get_title_from_url(self, suburl, headline):
        nameroot = suburl.split("/")[4]
        pattern = rf'{nameroot}/([^"]+?)/#'
        match = re.findall(pattern, suburl)
        if match == []:
            return headline
        else:
            return match[0].replace("-"," ").replace("/"," - ")

    def _create_document(self, paragraph_content, name, suburl, alternateName, description, titlefromurl):
        return Document(
            page_content=paragraph_content,
            metadata={
                "med_name": name,
                "url": suburl,
                "alternate_names": alternateName,
                "page_description": description,
                "document_description": titlefromurl,
            }
        )

    def _save_markdown(self, name, whole_page):
        mdname = f"testdata/NHSmed/{name}.md"
        with open(mdname, 'w', encoding='utf-8') as md_file:
            md_file.write(whole_page)
            print(f"Markdown created for {name}")

    def _save_json(self, name, documentjson):
        file_name = f"testdata/NHSmed/{name}.json"
        with open(file_name, 'w') as json_file:
            json.dump(documentjson, json_file)
            print(f"Document JSON created for {name}")

def main():
    api = NHSMedicationAPI()
    api.get_medication_list()
    medication_table = api.load_med_list()
    api.get_all_medications(medication_table)

if __name__ == "__main__":
    main()