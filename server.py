import os
import json
import time
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Set

from fastapi import FastAPI, HTTPException, Query, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field
import uvicorn
from fastapi.encoders import jsonable_encoder

from dsl_parser import DSLParser, clean_dsl_markup, normalize_headword, convert_dsl_to_html

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Custom JSONResponse with option to not encode Unicode
class ReadableJSONResponse(JSONResponse):
    def render(self, content: Any) -> bytes:
        # First, convert the content to a Python dictionary
        # This handles Pydantic models and other custom objects
        from fastapi.encoders import jsonable_encoder
        
        # Convert to JSON-compatible dict first
        json_compatible_content = jsonable_encoder(content)
        
        # Convert to JSON string and back to Python objects to ensure we're working with basic types
        json_str = json.dumps(json_compatible_content)
        python_dict = json.loads(json_str)
        
        # Now apply our deep cleaning to remove nulls and empty containers
        def deep_clean(obj):
            if obj is None:
                return None
            elif isinstance(obj, dict):
                # Create a new dict with non-null values and non-empty containers
                result = {}
                for k, v in obj.items():
                    cleaned_v = deep_clean(v)
                    if cleaned_v is not None:
                        if not isinstance(cleaned_v, (dict, list)) or len(cleaned_v) > 0:
                            result[k] = cleaned_v
                return result
            elif isinstance(obj, list):
                # Create a new list with non-null values and non-empty containers
                result = []
                for item in obj:
                    cleaned_item = deep_clean(item)
                    if cleaned_item is not None:
                        if not isinstance(cleaned_item, (dict, list)) or len(cleaned_item) > 0:
                            result.append(cleaned_item)
                return result
            else:
                # Return other types as is
                return obj
        
        # Apply deep cleaning to the Python dict
        cleaned_content = deep_clean(python_dict)
        
        # Serialize to JSON
        return json.dumps(
            cleaned_content,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            separators=(",", ": "),
        ).encode("utf-8")

# Create FastAPI app
app = FastAPI(
    title="Dictionary API",
    description="""
    OpenAPI server for dictionary lookups with comprehensive linguistic data.
    
    ## Dictionary Data Format
    
    The dictionary entries are stored in DSL (Dictionary Specification Language) format, which uses special markup tags for formatting and semantic information. 
    
    ### Options for handling DSL markup:
    
    1. **clean_markup=true** (default): Removes DSL markup tags and converts them to more readable text
    2. **structured=true**: Parses the DSL markup into structured JSON data with separate fields for different components
    3. **clean_markup=false**: Returns the raw DSL markup, useful for specialized applications
    4. **html_output=true**: Converts DSL markup to HTML for better readability in browsers and JSON viewers
    
    ### Common DSL tags you might see with clean_markup=false:
    
    - `[p]` - Paragraph
    - `[i]` - Italic text
    - `[b]` - Bold text
    - `[c]` - Color/formatting control
    - `[trn]` - Translation (may be numbered like `[trn1]`)
    - `[com]` - Comment (typically contains grammatical information)
    - `[m]` - Meaning (may be numbered)
    - `[ex]` - Example usage
    
    For most applications, using `clean_markup=true`, `structured=true`, or `html_output=true` is recommended.
    """,
    version="1.0.0",
)

# Add CORS middleware to allow cross-origin requests if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Path to dictionary files and cache
DICT_DIR = Path("Dicts")
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

# Store for parsed dictionaries (will be populated on startup)
dictionaries = {}
dictionary_info = {}

# Inverted index for faster word lookup across dictionaries
inverted_index = {}

# Helper function to safely get display names
def get_display_name(dict_name):
    """Helper function to get the display name of a dictionary safely"""
    # Check dictionary_info for display_name
    if dict_name in dictionary_info and "display_name" in dictionary_info[dict_name]:
        return dictionary_info[dict_name]["display_name"]
    
    return dict_name  # Fallback to the dictionary name if display_name is not available

# Response models for API
class DictionaryInfo(BaseModel):
    name: str = Field(..., description="Dictionary name")
    display_name: str = Field(..., description="Human-readable dictionary name")
    entry_count: int = Field(..., description="Number of entries in the dictionary")
    description: Optional[str] = Field(None, description="Dictionary description")
    language_from: str = Field(..., description="Source language")
    language_to: Optional[str] = Field(None, description="Target language for bilingual dictionaries")
    file_size: int = Field(..., description="Dictionary file size in bytes")

class DictionaryStats(BaseModel):
    total_entries: int = Field(..., description="Total number of entries across all dictionaries")
    total_size_bytes: int = Field(..., description="Total size of all dictionaries in bytes")
    average_entries_per_dictionary: float = Field(..., description="Average number of entries per dictionary")
    largest_dictionary: Optional[str] = Field(None, description="Name of the dictionary with the most entries")
    smallest_dictionary: Optional[str] = Field(None, description="Name of the dictionary with the fewest entries")

class DictionaryList(BaseModel):
    dictionaries: List[DictionaryInfo] = Field(..., description="List of available dictionaries")
    total_count: int = Field(..., description="Total number of dictionaries")
    statistics: Optional[DictionaryStats] = Field(None, description="Additional statistics about the dictionaries")
    
    class Config:
        exclude_none = True

class PartOfSpeech(BaseModel):
    value: str = Field(..., description="Part of speech value (e.g., noun, verb)")
    abbreviation: Optional[str] = Field(None, description="Abbreviation (e.g., n., v.)")

class Translation(BaseModel):
    text: str = Field(..., description="Translation text")
    examples: Optional[List[str]] = Field(None, description="Usage examples")
    
    class Config:
        exclude_none = True

class Meaning(BaseModel):
    number: Optional[int] = Field(None, description="Meaning number for words with multiple meanings")
    definition: Optional[str] = Field(None, description="Definition text")
    look_for: Optional[List[str]] = Field(None, description="Words or phrases to look up as cross-references")
    translations: Optional[List[Translation]] = Field(None, description="Translations")
    examples: Optional[List[str]] = Field(None, description="Usage examples")
    
    class Config:
        exclude_none = True
    
    def dict(self, *args, **kwargs):
        """Override dict method to exclude null values"""
        kwargs["exclude_none"] = True
        result = super().dict(*args, **kwargs)
        # Further filter out null values
        return {k: v for k, v in result.items() if v is not None}

class StructuredWordEntry(BaseModel):
    word: str = Field(..., description="The word or phrase")
    dictionary: str = Field(..., description="Source dictionary name")
    dictionary_display_name: str = Field(..., description="Human-readable dictionary name")
    part_of_speech: Optional[List[PartOfSpeech]] = Field(None, description="Parts of speech")
    meanings: Optional[List[Meaning]] = Field(None, description="Word meanings")
    pronunciation: Optional[str] = Field(None, description="Pronunciation")
    etymology: Optional[str] = Field(None, description="Etymology information")
    html_definition: str = Field(..., description="HTML-formatted definition with all linguistic data")
    
    class Config:
        exclude_none = True

class WordEntry(BaseModel):
    word: str = Field(..., description="The word or phrase")
    dictionary: str = Field(..., description="Source dictionary name")
    dictionary_display_name: str = Field(..., description="Human-readable dictionary name")
    definition: Any = Field(..., description="Complete definition with all linguistic data")
    
    class Config:
        exclude_none = True

class WordLookupResponse(BaseModel):
    entries: List[Union[WordEntry, StructuredWordEntry]] = Field(..., description="List of matching entries")
    query: str = Field(..., description="Original query")
    count: int = Field(..., description="Number of entries found")
    dictionaries_searched: List[str] = Field(..., description="Names of dictionaries searched")
    time_taken: float = Field(..., description="Time taken for the lookup in seconds")
    
    class Config:
        exclude_none = True

class PaginatedEntriesResponse(BaseModel):
    entries: List[Union[WordEntry, StructuredWordEntry]] = Field(..., description="List of entries")
    total_count: int = Field(..., description="Total number of entries")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Number of entries per page")
    has_next: bool = Field(..., description="Whether there are more entries to fetch")
    dictionary: Optional[str] = Field(None, description="Dictionary that was queried")
    time_taken: float = Field(..., description="Time taken for the lookup in seconds")
    
    class Config:
        exclude_none = True

# Middleware to measure response time
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Custom OpenAPI schema to ensure proper documentation
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    
    # Customize OpenAPI schema here if needed
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Create an inverted index for faster lookup
def build_inverted_index(all_dictionaries: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, Set[str]]]:
    """Build an inverted index mapping normalized words to dictionaries and original headwords"""
    index = {}
    
    for dict_name, entries in all_dictionaries.items():
        for headword in entries:
            # Use normalized form for lookup
            norm_word = normalize_headword(headword)
            
            if norm_word not in index:
                index[norm_word] = {}
            
            if dict_name not in index[norm_word]:
                index[norm_word][dict_name] = set()
            
            index[norm_word][dict_name].add(headword)
    
    return index

def parse_dsl_structure(definition: str) -> Dict:
    """Parse DSL markup into structured data"""
    # Extract part of speech (usually in [p]...[/p] or [pos]...[/pos] tags)
    pos_list = []
    pos_matches = re.finditer(r'\[p\](.*?)\[/p\]|\[pos\](.*?)\[/pos\]', definition)
    for match in pos_matches:
        pos_value = match.group(1) if match.group(1) else match.group(2)
        # Clean markup from pos_value
        pos_value = clean_dsl_markup(pos_value)
        # Extract abbreviation if in parentheses
        abbr_match = re.search(r'\((.*?)\)', pos_value)
        abbreviation = abbr_match.group(1) if abbr_match else None
        pos_list.append(PartOfSpeech(value=pos_value, abbreviation=abbreviation))
    
    # Extract meanings (usually numbered items or [m]...[/m] tags)
    meanings = []
    # Try to find numbered meanings
    meaning_matches = re.finditer(r'(\d+\.\s*)(.*?)(?=\d+\.\s*|$)', definition, re.DOTALL)
    
    if not list(re.finditer(r'\d+\.\s*', definition)):  # If no numbered items
        # Try to extract from [m] tags or just take the whole definition
        meaning_matches = re.finditer(r'\[m\d*\](.*?)\[/m\]', definition)
        if not list(re.finditer(r'\[m\d*\]', definition)):
            # Just use the whole definition as a single meaning
            meaning_text = re.sub(r'\[p\].*?\[/p\]|\[pos\].*?\[/pos\]', '', definition)
            # Clean markup from meaning_text
            clean_meaning_text = clean_dsl_markup(meaning_text.strip())
            meanings.append(Meaning(definition=clean_meaning_text))
    
    for idx, match in enumerate(meaning_matches):
        meaning_number = idx + 1
        meaning_text = match.group(2) if len(match.groups()) > 1 else match.group(1)
        
        # Extract translations if any (usually in [trn]...[/trn] tags)
        translations = []
        trn_matches = re.finditer(r'\[trn\](.*?)\[/trn\]', meaning_text)
        for trn_match in trn_matches:
            # Clean markup from translation text
            clean_trn_text = clean_dsl_markup(trn_match.group(1))
            translations.append(Translation(text=clean_trn_text))
        
        # Extract examples if any (usually in [ex]...[/ex] tags)
        examples = []
        ex_matches = re.finditer(r'\[ex\](.*?)\[/ex\]', meaning_text)
        for ex_match in ex_matches:
            # Clean markup from example text
            clean_ex_text = clean_dsl_markup(ex_match.group(1))
            examples.append(clean_ex_text)
        
        # Clean up the meaning text
        clean_meaning = re.sub(r'\[trn\].*?\[/trn\]|\[ex\].*?\[/ex\]', '', meaning_text)
        # Clean markup from clean_meaning
        clean_meaning = clean_dsl_markup(clean_meaning.strip())
        
        # Check for cross-references (e.g., "See: word")
        look_for = []
        see_match = re.search(r'^(?:-\s*)?(?:See|См\.?|Смотри(?:те)?):?\s+(.*?)$', clean_meaning, re.IGNORECASE)
        if see_match:
            references = see_match.group(1).split(',')
            look_for = [ref.strip() for ref in references]
            clean_meaning = None  # Set definition to None when it's only a cross-reference
        
        # Also check for [ref] tags in the original meaning text
        ref_matches = re.finditer(r'\[ref\](.*?)\[/ref\]', meaning_text)
        for ref_match in ref_matches:
            ref_word = ref_match.group(1).strip()
            if ref_word not in look_for:  # Avoid duplicates
                look_for.append(ref_word)
        
        meanings.append(Meaning(
            number=meaning_number,
            definition=clean_meaning,
            look_for=look_for if look_for else None,
            translations=translations if translations else None,
            examples=examples if examples else None
        ))
    
    # Extract pronunciation (usually in [pr]...[/pr] tags)
    pronunciation = None
    pr_match = re.search(r'\[pr\](.*?)\[/pr\]', definition)
    if pr_match:
        # Clean markup from pronunciation
        pronunciation = clean_dsl_markup(pr_match.group(1))
    
    # Extract etymology if present (usually after "From" or in [etym]...[/etym] tags)
    etymology = None
    etym_match = re.search(r'\[etym\](.*?)\[/etym\]', definition)
    if etym_match:
        # Clean markup from etymology
        etymology = clean_dsl_markup(etym_match.group(1))
    else:
        # Try to find lines starting with "From" which often indicate etymology
        etym_match = re.search(r'(?:^|\n)(?:From|От|Из) (.*?)(?:$|\n)', definition)
        if etym_match:
            # Clean markup from etymology
            etymology = clean_dsl_markup(etym_match.group(1))
    
    # Create a result dictionary without null values
    result = {}
    
    if pos_list:
        result["part_of_speech"] = pos_list
    
    if meanings:
        # Filter out null values from meanings
        cleaned_meanings = []
        for meaning in meanings:
            # Convert meaning to dict for easier manipulation
            meaning_dict = meaning.dict()
            # Remove null values
            meaning_dict = {k: v for k, v in meaning_dict.items() if v is not None}
            # Convert back to Meaning object
            cleaned_meanings.append(Meaning(**meaning_dict))
        
        result["meanings"] = cleaned_meanings
    
    if pronunciation:
        result["pronunciation"] = pronunciation
    
    if etymology:
        result["etymology"] = etymology
    
    # Always include html_definition with original DSL markup
    result["html_definition"] = definition
    
    return result

# Startup event - load dictionaries
@app.on_event("startup")
async def startup_event():
    global dictionaries, dictionary_info, inverted_index
    
    logger.info("Starting Dictionary API server...")
    
    # Check if we have cached JSON files
    if (CACHE_DIR / "dictionary_info.json").exists():
        logger.info("Loading dictionaries from cache...")
        dictionary_info, dictionaries = DSLParser.load_from_json(CACHE_DIR)
    else:
        # Parse DSL files
        logger.info("Parsing DSL dictionaries...")
        parser = DSLParser(DICT_DIR)
        dictionary_info, dictionaries = parser.parse_all_dictionaries()
        
        # Save to cache for faster loading next time
        parser.save_to_json(CACHE_DIR)
    
    # Build inverted index for faster lookups
    logger.info("Building inverted index for faster lookups...")
    inverted_index = build_inverted_index(dictionaries)
    
    logger.info(f"Loaded {len(dictionaries)} dictionaries with {sum(info['entry_count'] for info in dictionary_info.values())} total entries")

# Dictionary filtering dependency
def get_dictionary_filter(dict_names: Optional[List[str]] = Query(None, description="Filter results to specific dictionaries")):
    """Filter for dictionaries to search in"""
    if dict_names is None or len(dict_names) == 0:
        return None
    return set(dict_names)

# Endpoints
@app.get("/dictionaries", response_model=DictionaryList, tags=["Dictionaries"])
async def list_dictionaries():
    """
    List all available dictionaries with their statistics
    """
    try:
        if not dictionary_info:
            raise HTTPException(status_code=500, detail="Dictionaries not loaded")
        
        logger.info(f"Creating dictionary list with {len(dictionary_info)} dictionaries")
        dict_list = []
        for info in dictionary_info.values():
            # Ensure all required fields are present
            if "display_name" not in info:
                info["display_name"] = info["name"]
            dict_list.append(DictionaryInfo(**info))
        
        # Calculate additional statistics
        total_entries = sum(info.entry_count for info in dict_list)
        total_size = sum(info.file_size for info in dict_list)
        avg_entries = total_entries / len(dict_list) if dict_list else 0
        
        # Find largest and smallest dictionaries
        largest_dict = None
        smallest_dict = None
        if dict_list:
            largest_dict = max(dict_list, key=lambda x: x.entry_count).display_name
            smallest_dict = min(dict_list, key=lambda x: x.entry_count).display_name
        
        # Create the response with statistics
        response = DictionaryList(
            dictionaries=dict_list,
            total_count=len(dict_list),
            statistics=DictionaryStats(
                total_entries=total_entries,
                total_size_bytes=total_size,
                average_entries_per_dictionary=round(avg_entries, 2),
                largest_dictionary=largest_dict,
                smallest_dictionary=smallest_dict
            )
        )
        
        logger.info(f"Successfully created dictionary list with {len(dict_list)} dictionaries")
        return response
    except Exception as e:
        logger.error(f"Error in list_dictionaries: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing dictionaries: {str(e)}")

@app.get("/lookup/{word}", response_model=WordLookupResponse, tags=["Dictionary Lookup"])
async def lookup_word(
    word: str,
    dict_filter: Optional[Set[str]] = Depends(get_dictionary_filter),
    exact_match: bool = Query(True, description="Whether to perform exact matching or allow partial matches"),
    clean_markup: bool = Query(True, description="Whether to clean DSL markup tags (like [p], [trn], [com]) in definitions by removing them or converting to simple text. Set to false to see the raw dictionary format."),
    structured: bool = Query(True, description="Whether to return structured data with parsed DSL markup into separate fields for meanings, translations, etc. Default is True for structured output, set to False for simple format."),
    readable_text: bool = Query(False, description="Whether to return readable Unicode text instead of escaped characters (helpful for non-Latin alphabets)"),
    html_output: bool = Query(False, description="Whether to convert DSL markup to HTML for better display in browsers and JSON viewers. When set to true, this takes precedence over clean_markup."),
    include_references: bool = Query(True, description="Whether to include entries referenced in look_for sections (non-recursive)")
):
    """
    Look up a word or phrase in the dictionaries
    """
    start_time = time.time()
    
    # Normalize the word for lookup
    norm_word = normalize_headword(word)
    entries = []
    dicts_searched = set()
    
    # If exact match, look up in inverted index
    if exact_match:
        if norm_word in inverted_index:
            for dict_name, headwords in inverted_index[norm_word].items():
                # Apply dictionary filter if provided
                if dict_filter and dict_name not in dict_filter:
                    continue
                
                dicts_searched.add(dict_name)
                
                for original_headword in headwords:
                    definition = dictionaries[dict_name][original_headword]
                    
                    if structured:
                        # Parse structure before cleaning markup
                        structure_data = parse_dsl_structure(definition)
                        
                        # Always convert the html_definition field to HTML
                        structure_data["html_definition"] = convert_dsl_to_html(definition)
                            
                        entries.append(StructuredWordEntry(
                            word=original_headword,
                            dictionary=dict_name,
                            dictionary_display_name=get_display_name(dict_name),
                            **structure_data
                        ))
                    else:
                        if html_output:
                            definition = convert_dsl_to_html(definition)
                        elif clean_markup:
                            definition = clean_dsl_markup(definition)
                        
                        entries.append(WordEntry(
                            word=original_headword,
                            dictionary=dict_name,
                            dictionary_display_name=get_display_name(dict_name),
                            definition=definition
                        ))
    else:
        # Partial match - look for words that contain the query
        for dict_name, dict_entries in dictionaries.items():
            # Apply dictionary filter if provided
            if dict_filter and dict_name not in dict_filter:
                continue
            
            dicts_searched.add(dict_name)
            
            for headword, definition in dict_entries.items():
                if norm_word in normalize_headword(headword):
                    if structured:
                        # Parse structure before cleaning markup
                        structure_data = parse_dsl_structure(definition)
                        
                        # Always convert the html_definition field to HTML
                        structure_data["html_definition"] = convert_dsl_to_html(definition)
                            
                        entries.append(StructuredWordEntry(
                            word=headword,
                            dictionary=dict_name,
                            dictionary_display_name=get_display_name(dict_name),
                            **structure_data
                        ))
                    else:
                        if html_output:
                            definition = convert_dsl_to_html(definition)
                        elif clean_markup:
                            definition = clean_dsl_markup(definition)
                        
                        entries.append(WordEntry(
                            word=headword,
                            dictionary=dict_name,
                            dictionary_display_name=get_display_name(dict_name),
                            definition=definition
                        ))
    
    # Check if we found anything
    if not entries:
        # No exact match, try case-insensitive search
        for dict_name, dict_entries in dictionaries.items():
            # Apply dictionary filter if provided
            if dict_filter and dict_name not in dict_filter:
                continue
            
            dicts_searched.add(dict_name)
            
            for headword, definition in dict_entries.items():
                norm_headword = normalize_headword(headword)
                if norm_word == norm_headword:
                    if structured:
                        # Parse structure before cleaning markup
                        structure_data = parse_dsl_structure(definition)
                        
                        # Always convert the html_definition field to HTML
                        structure_data["html_definition"] = convert_dsl_to_html(definition)
                            
                        entries.append(StructuredWordEntry(
                            word=headword,
                            dictionary=dict_name,
                            dictionary_display_name=get_display_name(dict_name),
                            **structure_data
                        ))
                    else:
                        if html_output:
                            definition = convert_dsl_to_html(definition)
                        elif clean_markup:
                            definition = clean_dsl_markup(definition)
                        
                        entries.append(WordEntry(
                            word=headword,
                            dictionary=dict_name,
                            dictionary_display_name=get_display_name(dict_name),
                            definition=definition
                        ))
    
    # If still nothing found, return 404
    if not entries:
        raise HTTPException(status_code=404, detail=f"No entries found for '{word}'")
    
    # Look up referenced entries if requested
    referenced_entries = []
    if include_references and structured:
        referenced_words = set()
        
        # Collect all look_for words
        for entry in entries:
            if isinstance(entry, StructuredWordEntry) and entry.meanings:
                for meaning in entry.meanings:
                    if meaning.look_for:
                        for ref_word in meaning.look_for:
                            # Don't add self-references
                            if normalize_headword(ref_word) != norm_word:
                                referenced_words.add(ref_word)
        
        # Look up each referenced word
        for ref_word in referenced_words:
            # Normalize the word for lookup
            norm_ref_word = normalize_headword(ref_word)
            
            # If in inverted index, look it up
            if norm_ref_word in inverted_index:
                for dict_name, headwords in inverted_index[norm_ref_word].items():
                    # Apply dictionary filter if provided
                    if dict_filter and dict_name not in dict_filter:
                        continue
                    
                    dicts_searched.add(dict_name)
                    
                    for original_headword in headwords:
                        definition = dictionaries[dict_name][original_headword]
                        
                        if structured:
                            # Parse structure before cleaning markup
                            structure_data = parse_dsl_structure(definition)
                            
                            # Always convert the html_definition field to HTML
                            structure_data["html_definition"] = convert_dsl_to_html(definition)
                                
                            referenced_entries.append(StructuredWordEntry(
                                word=original_headword,
                                dictionary=dict_name,
                                dictionary_display_name=get_display_name(dict_name),
                                **structure_data
                            ))
                        else:
                            if html_output:
                                definition = convert_dsl_to_html(definition)
                            elif clean_markup:
                                definition = clean_dsl_markup(definition)
                            
                            referenced_entries.append(WordEntry(
                                word=original_headword,
                                dictionary=dict_name,
                                dictionary_display_name=get_display_name(dict_name),
                                definition=definition
                            ))
        
        # Add referenced entries to the result
        entries.extend(referenced_entries)
    
    response_data = WordLookupResponse(
        entries=entries,
        query=word,
        count=len(entries),
        dictionaries_searched=list(dicts_searched),
        time_taken=time.time() - start_time
    )
    
    # Use a custom response class that will exclude null values
    return ReadableJSONResponse(content=response_data)

@app.get("/multi-lookup", response_model=WordLookupResponse, tags=["Dictionary Lookup"])
async def multi_word_lookup(
    query: str = Query(..., description="Space-separated words to look up"),
    dict_filter: Optional[Set[str]] = Depends(get_dictionary_filter),
    clean_markup: bool = Query(True, description="Whether to clean DSL markup tags (like [p], [trn], [com]) in definitions by removing them or converting to simple text"),
    structured: bool = Query(True, description="Whether to return structured data with parsed DSL markup into separate fields for meanings, translations, etc. Default is True for structured output, set to False for simple format."),
    readable_text: bool = Query(False, description="Whether to return readable Unicode text instead of escaped characters (helpful for non-Latin alphabets)"),
    html_output: bool = Query(False, description="Whether to convert DSL markup to HTML for better display in browsers and JSON viewers. When set to true, this takes precedence over clean_markup."),
    include_references: bool = Query(True, description="Whether to include entries referenced in look_for sections (non-recursive)")
):
    """
    Look up multiple words or phrases in the dictionaries.
    Returns complete articles for each word.
    """
    start_time = time.time()
    words = query.split()
    entries = []
    dicts_searched = set()
    
    for word in words:
        # Normalize the word for lookup
        norm_word = normalize_headword(word)
        
        word_found = False
        
        # Look up in inverted index
        if norm_word in inverted_index:
            for dict_name, headwords in inverted_index[norm_word].items():
                # Apply dictionary filter if provided
                if dict_filter and dict_name not in dict_filter:
                    continue
                
                dicts_searched.add(dict_name)
                
                for original_headword in headwords:
                    word_found = True
                    definition = dictionaries[dict_name][original_headword]
                    
                    if structured:
                        # Parse structure before cleaning markup
                        structure_data = parse_dsl_structure(definition)
                        
                        # Always convert the html_definition field to HTML
                        structure_data["html_definition"] = convert_dsl_to_html(definition)
                            
                        entries.append(StructuredWordEntry(
                            word=original_headword,
                            dictionary=dict_name,
                            dictionary_display_name=get_display_name(dict_name),
                            **structure_data
                        ))
                    else:
                        if html_output:
                            definition = convert_dsl_to_html(definition)
                        elif clean_markup:
                            definition = clean_dsl_markup(definition)
                        
                        entries.append(WordEntry(
                            word=original_headword,
                            dictionary=dict_name,
                            dictionary_display_name=get_display_name(dict_name),
                            definition=definition
                        ))
    
    # If no entries found, return 404
    if not entries:
        raise HTTPException(status_code=404, detail=f"No entries found for '{query}'")
    
    # Look up referenced entries if requested
    referenced_entries = []
    if include_references and structured:
        referenced_words = set()
        
        # Collect all look_for words
        for entry in entries:
            if isinstance(entry, StructuredWordEntry) and entry.meanings:
                for meaning in entry.meanings:
                    if meaning.look_for:
                        for ref_word in meaning.look_for:
                            # Don't add self-references
                            if normalize_headword(ref_word) not in [normalize_headword(w) for w in words]:
                                referenced_words.add(ref_word)
        
        # Look up each referenced word
        for ref_word in referenced_words:
            # Normalize the word for lookup
            norm_ref_word = normalize_headword(ref_word)
            
            # If in inverted index, look it up
            if norm_ref_word in inverted_index:
                for dict_name, headwords in inverted_index[norm_ref_word].items():
                    # Apply dictionary filter if provided
                    if dict_filter and dict_name not in dict_filter:
                        continue
                    
                    dicts_searched.add(dict_name)
                    
                    for original_headword in headwords:
                        definition = dictionaries[dict_name][original_headword]
                        
                        if structured:
                            # Parse structure before cleaning markup
                            structure_data = parse_dsl_structure(definition)
                            
                            # Always convert the html_definition field to HTML
                            structure_data["html_definition"] = convert_dsl_to_html(definition)
                                
                            referenced_entries.append(StructuredWordEntry(
                                word=original_headword,
                                dictionary=dict_name,
                                dictionary_display_name=get_display_name(dict_name),
                                **structure_data
                            ))
                        else:
                            if html_output:
                                definition = convert_dsl_to_html(definition)
                            elif clean_markup:
                                definition = clean_dsl_markup(definition)
                            
                            referenced_entries.append(WordEntry(
                                word=original_headword,
                                dictionary=dict_name,
                                dictionary_display_name=get_display_name(dict_name),
                                definition=definition
                            ))
        
        # Add referenced entries to the result
        entries.extend(referenced_entries)
    
    response_data = WordLookupResponse(
        entries=entries,
        query=query,
        count=len(entries),
        dictionaries_searched=list(dicts_searched),
        time_taken=time.time() - start_time
    )
    
    # Return with readable text if requested
    if readable_text:
        return ReadableJSONResponse(content=response_data.dict())
    else:
        return response_data

@app.get("/search", response_model=WordLookupResponse, tags=["Dictionary Lookup"])
async def search_word(
    query: str = Query(..., description="Search query to find in dictionary entries"),
    dict_filter: Optional[Set[str]] = Depends(get_dictionary_filter),
    exact_match: bool = Query(False, description="Whether to perform exact word matching or search within definitions"),
    clean_markup: bool = Query(True, description="Whether to clean DSL markup tags (like [p], [trn], [com]) in definitions by removing them or converting to simple text"),
    structured: bool = Query(True, description="Whether to return structured data with parsed DSL markup into separate fields for meanings, translations, etc. Default is True for structured output, set to False for simple format."),
    limit: int = Query(10, description="Maximum number of entries to return"),
    readable_text: bool = Query(False, description="Whether to return readable Unicode text instead of escaped characters (helpful for non-Latin alphabets)"),
    html_output: bool = Query(False, description="Whether to convert DSL markup to HTML for better display in browsers and JSON viewers. When set to true, this takes precedence over clean_markup."),
    include_references: bool = Query(True, description="Whether to include entries referenced in look_for sections (non-recursive)")
):
    """
    Search for words or phrases across dictionaries.
    This can search in headwords or definitions based on the exact_match parameter.
    """
    start_time = time.time()
    
    # Normalize the query for case-insensitive search
    norm_query = normalize_headword(query)
    entries = []
    dicts_searched = set()
    
    # Exact headword match (similar to lookup but allows partial matching of headword)
    if exact_match:
        # Search through all dictionary headwords
        for dict_name, dict_entries in dictionaries.items():
            # Apply dictionary filter if provided
            if dict_filter and dict_name not in dict_filter:
                continue
            
            dicts_searched.add(dict_name)
            
            # Search through headwords
            for headword, definition in dict_entries.items():
                norm_headword = normalize_headword(headword)
                if norm_query in norm_headword:  # Partial match on headword
                    if structured:
                        # Parse structure before cleaning markup
                        structure_data = parse_dsl_structure(definition)
                        
                        # Always convert the html_definition field to HTML
                        structure_data["html_definition"] = convert_dsl_to_html(definition)
                            
                        entries.append(StructuredWordEntry(
                            word=headword,
                            dictionary=dict_name,
                            dictionary_display_name=get_display_name(dict_name),
                            **structure_data
                        ))
                    else:
                        if html_output:
                            definition = convert_dsl_to_html(definition)
                        elif clean_markup:
                            definition = clean_dsl_markup(definition)
                        
                        entries.append(WordEntry(
                            word=headword,
                            dictionary=dict_name,
                            dictionary_display_name=get_display_name(dict_name),
                            definition=definition
                        ))
                
                # Limit number of results
                if len(entries) >= limit:
                    break
            
            # Limit number of results
            if len(entries) >= limit:
                break
    else:
        # Full-text search in definitions
        for dict_name, dict_entries in dictionaries.items():
            # Apply dictionary filter if provided
            if dict_filter and dict_name not in dict_filter:
                continue
            
            dicts_searched.add(dict_name)
            
            # Search through definitions
            for headword, definition in dict_entries.items():
                # Simple check if query is in definition (case-insensitive)
                if norm_query in normalize_headword(definition):
                    if structured:
                        # Parse structure before cleaning markup
                        structure_data = parse_dsl_structure(definition)
                        
                        # Always convert the html_definition field to HTML
                        structure_data["html_definition"] = convert_dsl_to_html(definition)
                            
                        entries.append(StructuredWordEntry(
                            word=headword,
                            dictionary=dict_name,
                            dictionary_display_name=get_display_name(dict_name),
                            **structure_data
                        ))
                    else:
                        if html_output:
                            definition = convert_dsl_to_html(definition)
                        elif clean_markup:
                            definition = clean_dsl_markup(definition)
                        
                        entries.append(WordEntry(
                            word=headword,
                            dictionary=dict_name,
                            dictionary_display_name=get_display_name(dict_name),
                            definition=definition
                        ))
                
                # Limit number of results
                if len(entries) >= limit:
                    break
            
            # Limit number of results
            if len(entries) >= limit:
                break
    
    # If no entries found, return 404
    if not entries:
        raise HTTPException(status_code=404, detail=f"No entries found for search query '{query}'")
    
    # Look up referenced entries if requested
    referenced_entries = []
    if include_references and structured:
        referenced_words = set()
        
        # Collect all look_for words
        for entry in entries:
            if isinstance(entry, StructuredWordEntry) and entry.meanings:
                for meaning in entry.meanings:
                    if meaning.look_for:
                        for ref_word in meaning.look_for:
                            # Don't add self-references
                            if normalize_headword(ref_word) != normalize_headword(query):
                                referenced_words.add(ref_word)
        
        # Look up each referenced word
        for ref_word in referenced_words:
            # Normalize the word for lookup
            norm_ref_word = normalize_headword(ref_word)
            
            # If in inverted index, look it up
            if norm_ref_word in inverted_index:
                for dict_name, headwords in inverted_index[norm_ref_word].items():
                    # Apply dictionary filter if provided
                    if dict_filter and dict_name not in dict_filter:
                        continue
                    
                    dicts_searched.add(dict_name)
                    
                    for original_headword in headwords:
                        definition = dictionaries[dict_name][original_headword]
                        
                        if structured:
                            # Parse structure before cleaning markup
                            structure_data = parse_dsl_structure(definition)
                            
                            # Always convert the html_definition field to HTML
                            structure_data["html_definition"] = convert_dsl_to_html(definition)
                                
                            referenced_entries.append(StructuredWordEntry(
                                word=original_headword,
                                dictionary=dict_name,
                                dictionary_display_name=get_display_name(dict_name),
                                **structure_data
                            ))
                        else:
                            if html_output:
                                definition = convert_dsl_to_html(definition)
                            elif clean_markup:
                                definition = clean_dsl_markup(definition)
                            
                            referenced_entries.append(WordEntry(
                                word=original_headword,
                                dictionary=dict_name,
                                dictionary_display_name=get_display_name(dict_name),
                                definition=definition
                            ))
        
        # Add referenced entries to the result
        entries.extend(referenced_entries)
    
    response_data = WordLookupResponse(
        entries=entries,
        query=query,
        count=len(entries),
        dictionaries_searched=list(dicts_searched),
        time_taken=time.time() - start_time
    )
    
    # Return with readable text if requested
    if readable_text:
        return ReadableJSONResponse(content=response_data.dict())
    else:
        return response_data

@app.get("/prefix-search/{prefix}", response_model=WordLookupResponse, tags=["Dictionary Lookup"])
async def prefix_search(
    prefix: str,
    dict_filter: Optional[Set[str]] = Depends(get_dictionary_filter),
    clean_markup: bool = Query(True, description="Whether to clean DSL markup tags (like [p], [trn], [com]) in definitions by removing them or converting to simple text"),
    structured: bool = Query(True, description="Whether to return structured data with parsed DSL markup into separate fields for meanings, translations, etc. Default is True for structured output, set to False for simple format."),
    limit: int = Query(10, description="Maximum number of entries to return"),
    readable_text: bool = Query(False, description="Whether to return readable Unicode text instead of escaped characters (helpful for non-Latin alphabets)"),
    html_output: bool = Query(False, description="Whether to convert DSL markup to HTML for better display in browsers and JSON viewers. When set to true, this takes precedence over clean_markup."),
):
    """
    Find dictionary entries that start with the given prefix
    """
    start_time = time.time()
    
    # Normalize the prefix for case-insensitive search
    norm_prefix = normalize_headword(prefix)
    entries = []
    dicts_searched = set()
    
    # Search in all dictionaries
    for dict_name, dict_entries in dictionaries.items():
        # Apply dictionary filter if provided
        if dict_filter and dict_name not in dict_filter:
            continue
        
        dicts_searched.add(dict_name)
        
        # Find words starting with the prefix
        for headword, definition in dict_entries.items():
            norm_headword = normalize_headword(headword)
            if norm_headword.startswith(norm_prefix):
                if structured:
                    # Parse structure before cleaning markup
                    structure_data = parse_dsl_structure(definition)
                    
                    # Always convert the html_definition field to HTML
                    structure_data["html_definition"] = convert_dsl_to_html(definition)
                        
                    entries.append(StructuredWordEntry(
                        word=headword,
                        dictionary=dict_name,
                        dictionary_display_name=get_display_name(dict_name),
                        **structure_data
                    ))
                else:
                    if html_output:
                        definition = convert_dsl_to_html(definition)
                    elif clean_markup:
                        definition = clean_dsl_markup(definition)
                    
                    entries.append(WordEntry(
                        word=headword,
                        dictionary=dict_name,
                        dictionary_display_name=get_display_name(dict_name),
                        definition=definition
                    ))
            
            # Limit number of results
            if len(entries) >= limit:
                break
        
        # Limit number of results
        if len(entries) >= limit:
            break
    
    # If no entries found, return 404
    if not entries:
        raise HTTPException(status_code=404, detail=f"No entries found starting with '{prefix}'")
    
    response_data = WordLookupResponse(
        entries=entries,
        query=prefix,
        count=len(entries),
        dictionaries_searched=list(dicts_searched),
        time_taken=time.time() - start_time
    )
    
    # Return with readable text if requested
    if readable_text:
        return ReadableJSONResponse(content=response_data.dict())
    else:
        return response_data

@app.get("/list-entries", response_model=PaginatedEntriesResponse, tags=["Dictionary Lookup"])
async def list_entries(
    dict_name: Optional[str] = Query(None, description="Dictionary to list entries from. If not provided, lists from all dictionaries."),
    page: int = Query(1, description="Page number for pagination", ge=1),
    page_size: int = Query(20, description="Number of entries per page", ge=1, le=100),
    clean_markup: bool = Query(True, description="Whether to clean DSL markup tags (like [p], [trn], [com]) in definitions by removing them or converting to simple text"),
    structured: bool = Query(True, description="Whether to return structured data with parsed DSL markup into separate fields for meanings, translations, etc. Default is True for structured output, set to False for simple format."),
    readable_text: bool = Query(False, description="Whether to return readable Unicode text instead of escaped characters (helpful for non-Latin alphabets)"),
    html_output: bool = Query(False, description="Whether to convert DSL markup to HTML for better display in browsers and JSON viewers. When set to true, this takes precedence over clean_markup."),
):
    """
    List entries from dictionaries with pagination support
    """
    start_time = time.time()
    
    # Validate dictionary name if provided
    if dict_name and dict_name not in dictionaries:
        raise HTTPException(status_code=404, detail=f"Dictionary '{dict_name}' not found")
    
    all_entries = []
    
    # Get all entries from the specified dictionary or all dictionaries
    if dict_name:
        dict_entries = dictionaries[dict_name]
        for headword, definition in dict_entries.items():
            if structured:
                # Parse structure before cleaning markup
                structure_data = parse_dsl_structure(definition)
                
                # Always convert the html_definition field to HTML
                structure_data["html_definition"] = convert_dsl_to_html(definition)
                    
                all_entries.append(StructuredWordEntry(
                    word=headword,
                    dictionary=dict_name,
                    dictionary_display_name=get_display_name(dict_name),
                    **structure_data
                ))
            else:
                if html_output:
                    definition = convert_dsl_to_html(definition)
                elif clean_markup:
                    definition = clean_dsl_markup(definition)
                
                all_entries.append(WordEntry(
                    word=headword,
                    dictionary=dict_name,
                    dictionary_display_name=get_display_name(dict_name),
                    definition=definition
                ))
    else:
        # Get entries from all dictionaries
        for dictionary_name, dict_entries in dictionaries.items():
            for headword, definition in dict_entries.items():
                if structured:
                    # Parse structure before cleaning markup
                    structure_data = parse_dsl_structure(definition)
                    
                    # Always convert the html_definition field to HTML
                    structure_data["html_definition"] = convert_dsl_to_html(definition)
                        
                    all_entries.append(StructuredWordEntry(
                        word=headword,
                        dictionary=dictionary_name,
                        dictionary_display_name=get_display_name(dictionary_name),
                        **structure_data
                    ))
                else:
                    if html_output:
                        definition = convert_dsl_to_html(definition)
                    elif clean_markup:
                        definition = clean_dsl_markup(definition)
                    
                    all_entries.append(WordEntry(
                        word=headword,
                        dictionary=dictionary_name,
                        dictionary_display_name=get_display_name(dictionary_name),
                        definition=definition
                    ))
    
    # Sort entries by word
    all_entries.sort(key=lambda entry: entry.word)
    
    # Calculate pagination
    total_entries = len(all_entries)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    
    # Slice entries for current page
    if start_idx >= total_entries:
        raise HTTPException(status_code=404, detail=f"Page {page} exceeds available entries")
    
    all_entries = all_entries[start_idx:end_idx]
    has_next = (page * page_size) < total_entries
    
    response_data = PaginatedEntriesResponse(
        entries=all_entries,
        total_count=total_entries,
        page=page,
        page_size=page_size,
        has_next=has_next,
        dictionary=dict_name,
        time_taken=time.time() - start_time
    )
    
    # Return with readable text if requested
    if readable_text:
        return ReadableJSONResponse(content=response_data.dict())
    else:
        return response_data

# Health check endpoint
@app.get("/health", tags=["System"])
async def health_check():
    """
    Health check endpoint to verify the service is running
    """
    return {
        "status": "ok",
        "dictionaries_loaded": len(dictionaries),
        "entries_total": sum(info['entry_count'] for info in dictionary_info.values()),
        "uptime": time.time() - app.state.start_time if hasattr(app.state, 'start_time') else None
    }

# Set server start time
@app.on_event("startup")
async def set_start_time():
    app.state.start_time = time.time()

# Root endpoint - redirect to docs
@app.get("/", include_in_schema=False)
async def root():
    """Redirect to API documentation"""
    return RedirectResponse(url="/docs")

# Custom JSON encoder that skips null values
class NullExcludingJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, "dict") and callable(obj.dict):
            # Handle Pydantic models
            return {k: v for k, v in obj.dict().items() if v is not None}
        return super().default(obj)

class ReadableJSONResponse(JSONResponse):
    def render(self, content: Any) -> bytes:
        # First, convert the content to a Python dictionary
        # This handles Pydantic models and other custom objects
        from fastapi.encoders import jsonable_encoder
        
        # Convert to JSON-compatible dict first
        json_compatible_content = jsonable_encoder(content)
        
        # Function to recursively remove null values and empty containers
        def remove_nulls_and_empty(obj):
            if obj is None:
                return None
            
            if isinstance(obj, dict):
                # Process dictionary
                result = {}
                for k, v in obj.items():
                    processed_v = remove_nulls_and_empty(v)
                    # Only include non-null values and non-empty containers
                    if processed_v is not None:
                        if not isinstance(processed_v, (dict, list)) or len(processed_v) > 0:
                            result[k] = processed_v
                return result
            
            elif isinstance(obj, list):
                # Process list
                result = []
                for item in obj:
                    processed_item = remove_nulls_and_empty(item)
                    # Only include non-null values and non-empty containers
                    if processed_item is not None:
                        if not isinstance(processed_item, (dict, list)) or len(processed_item) > 0:
                            result.append(processed_item)
                return result
            
            else:
                # Return other types as is
                return obj
        
        # Apply cleaning to remove nulls and empty containers
        cleaned_content = remove_nulls_and_empty(json_compatible_content)
        
        # Serialize to JSON with custom encoder
        return json.dumps(
            cleaned_content,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            separators=(",", ": "),
            cls=NullExcludingJSONEncoder,
        ).encode("utf-8")

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
