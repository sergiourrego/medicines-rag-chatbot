import sys, os, string, time, requests, markdownify, re, json
from langchain.docstore.document import Document
from typing import Iterable

# Check for API Key
if "NHS_API_KEY" in os.environ:
    NHS_API_KEY = os.environ["NHS_API_KEY"]
else:
    print("NHS API Key missing from .env")
    sys.exit
    
base_url = "https://api.nhs.uk/medicines"
base_params = {
    "subscription-key": NHS_API_KEY,
}

# Check date script last run
if "DATE_LAST_RUN" in os.environ:
    DATE_LAST_RUN = os.environ["DATE_LAST_RUN"]
    
# Initial grab - all medication API urls and date modified
# DATE_LAST_RUN not set:
medication_table = {"data": []}
if "DATE_LAST_RUN" not in dir():
    # Grab in alphabetical order
    for letter in list(string.ascii_uppercase):
        try:
            base_params["category"] = letter
            response = requests.get(base_url, params=base_params)
            response.raise_for_status()  # Raise an exception for non-2xx status codes
            results = response.json()
            for object in results["significantLink"]:
                name, url, dateModified = object["name"], object["url"], object["mainEntityOfPage"]["dateModified"]
                medication_table["data"].append({"name": name, "url": url, "dateModified": dateModified})
                print(f"Drug: {name}, URL: {url}, Date Modified: {dateModified}")
        except requests.exceptions.RequestException as e:
            print(f"Error occurred while fetching medication list: {e}")
        except ValueError as e:
            print(f"Error occurred while parsing JSON response for medication list: {e}")
        except Exception as e:
            print(f"An unexpected error occurred for API request to medication list: {e}")
        # wait as trial subscription rate limit 10/min
        time.sleep(7)
    # save as JSON
    with open('backend/testdata/NHSmed/medication_table.json', 'w', encoding='utf-8') as json_file:
        json.dump(medication_table, json_file, ensure_ascii=False, indent=4)
else:
    # feat: use "orderBy: dateModified" and only grab recently updated links
    print(f"Script run on {DATE_LAST_RUN}. Only updating pages after this date")

# Load medication table JSON
with open('backend/testdata/NHSmed/medication_table.json', 'r', encoding='utf-8') as json_file:
  # Load the JSON data using json.load()
  medication_table = json.load(json_file)

# Load existing documents.json into dict
file_name = f"backend/testdata/NHSmed/documents.json"
try:
  with open(file_name, 'r') as json_file:
    documents = json.load(json_file)
except FileNotFoundError:
  documents = {}  # Empty dic if the file doesn't exist

# Grab data from each medication url
# Save to md and JSON file for LangChain Documents
for med in medication_table["data"]:
    name, url = med["name"], med["url"]
    try:
        response = requests.get(url, params=base_params)
        response.raise_for_status()  # Raise an exception for non-2xx status codes
        results = response.json()
        # Assign results to variable
        name, description, url = results['name'],  results['description'], results['url']
        # deconstruct array
        alternateName = results['about']['alternateName']
        alternateName = " ".join(alternateName)
        # page headings +- alternateName
        if alternateName == "":
            page_content = f"# {name}\n\n## {description}\n\n"
        else:
            page_content = f"# {name} ({alternateName})\n\n## {description}\n\n"
        # Collate all text in md file
        # feat: need to add subheading url as metadata. consider seperate docs for each subheading or other way to 
        for section in results['hasPart']:
            #url = object["url"]
            subdescription = section.get("description", "")
            headline = section.get("headline", "")
            ## add headline as subheading if present otherwise just description
            if headline == "":
                page_content += f"**{subdescription}**\n\n"
            else :
                page_content += f"## {headline}\n\n**{subdescription}**\n\n"
            ## add each paragraph
            for paragraph in section["hasPart"]:
                # Question blocks which need parsing differently
                if paragraph.get("@type") == "Question":
                    text = paragraph["acceptedAnswer"].get("text", "")
                    subhead = paragraph.get("name", "")
                else:
                    subhead = paragraph.get("headline","")
                    text = paragraph.get("text", "")
                # convert html to md
                mdtext = markdownify.markdownify(text)
                # remove links
                cleaned_md = re.sub(r'\[(.*?)\](.*?\))', r'\1', mdtext)
                # add subhead if present
                if subhead == "":
                    page_content += f"{cleaned_md}\n\n"
                else :
                    page_content += f"### {subhead}\n\n{cleaned_md}\n\n"
        # Save to md
        mdname = f"backend/testdata/NHSmed/{name}.md"
        with open(mdname, 'w', encoding='utf-8') as md_file:
            md_file.write(page_content)
            print(f"Markdown created for {name}")
        # Create LangChain Document and update document dict
        doc = Document(
            page_content=page_content,
            metadata={
                "name": name,
                "url": url,
                "alternateName": alternateName,
                "descripton": description
            }
        )
        documents[name] = doc.json()
        print(f"Document created for {name}")
            
    except requests.exceptions.RequestException as e:
        print(f"Error occurred while fetching data for {med}: {e}")
    except ValueError as e:
        print(f"Error occurred while parsing JSON response for {med}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred for {med}: {e}")
    # timeout
    time.sleep(7) 
# Write document dict to json
with open(file_name, "w") as json_file:
    json.dump(documents, json_file)
    print(f"Documents updated in documents.json")
    
# # Read to array of docs
# file_name = f"backend/testdata/NHSmed/documents.json"
# docarray = []
# with open(file_name, 'r') as json_file:
#     dict = json.load(json_file)
#     for object in dict.values():
#         obj = json.loads(object)
#         doc = Document(**obj)
#         docarray.append(doc)
