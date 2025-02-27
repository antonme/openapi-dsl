import re
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

logger = logging.getLogger(__name__)

class DSLParser:
    """Parser for DSL dictionary files"""
    
    def __init__(self, dict_dir: Path):
        self.dict_dir = dict_dir
        self.dictionaries = {}
        self.dictionary_info = {}
    
    def find_dsl_files(self) -> List[Path]:
        """Find all DSL files in the dictionary directory"""
        return list(self.dict_dir.glob("*.dsl"))
    
    def parse_header(self, dsl_file: Path) -> Dict[str, Any]:
        """Parse the header of a DSL file to extract metadata"""
        file_stem = dsl_file.stem
        info = {
            "name": file_stem,
            "file_size": dsl_file.stat().st_size,
            "language_from": "",
            "language_to": None,
            "description": None,
            "entry_count": 0,
        }
        
        # Create a human-readable display name from the filename
        # Replace hyphens with arrows for language pairs and clean up underscores
        display_name = file_stem.replace("-", " → ").replace("_", " ")
        # Capitalize words
        display_name = " ".join(word.capitalize() for word in display_name.split())
        info["display_name"] = display_name
        
        # Try to find matching .ann file for additional metadata
        ann_file = dsl_file.with_suffix(".ann")
        if ann_file.exists():
            try:
                # Detect encoding - first try utf-16le, which is commonly used for .ann files
                encodings_to_try = ['utf-16le', 'utf-8', 'windows-1251', 'latin-1']
                
                # Try each encoding until one works
                for encoding in encodings_to_try:
                    try:
                        with open(ann_file, 'r', encoding=encoding) as f:
                            content = f.read()
                        
                        # If we got here, the encoding worked
                        logger.info(f"Successfully read {ann_file} with {encoding} encoding")
                        
                        # Store the first few lines (or all content if it's small) as annotation_text
                        # This will be used as description if no #DESCRIPTION directive is found
                        lines = content.splitlines()
                        annotation_text = "\n".join(lines[:20] if len(lines) > 20 else lines)
                        
                        # Reset description from annotation
                        description_from_directive = None
                        
                        # Process the content line by line
                        for line in lines:
                            if line.startswith("#NAME"):
                                # Use the #NAME from the annotation file as the display_name
                                name_value = line[6:].strip().strip('"')
                                info["display_name"] = name_value
                            elif line.startswith("#INDEX_LANGUAGE"):
                                info["language_from"] = line[16:].strip().strip('"')
                            elif line.startswith("#CONTENTS_LANGUAGE"):
                                info["language_to"] = line[19:].strip().strip('"')
                            elif line.startswith("#DESCRIPTION"):
                                description_from_directive = line[13:].strip().strip('"')
                        
                        # Use #DESCRIPTION directive value if found, otherwise use annotation text
                        if description_from_directive:
                            info["description"] = description_from_directive
                        else:
                            # Filter out lines that are directives or empty
                            description_lines = [line for line in lines 
                                               if not line.startswith('#') and line.strip()]
                            if description_lines:
                                # Use the first 10 lines as description
                                info["description"] = "\n".join(description_lines[:10])
                        
                        # No need to try other encodings if this one worked
                        break
                    except UnicodeDecodeError:
                        # Try the next encoding
                        continue
                    except Exception as e:
                        logger.warning(f"Error reading annotation file {ann_file} with {encoding} encoding: {e}")
                        continue
            except Exception as e:
                logger.warning(f"Error reading annotation file {ann_file}: {e}")
        
        # If language info is still missing, try to infer from filename
        if not info["language_from"]:
            name_parts = file_stem.split('-')
            if len(name_parts) > 1:
                info["language_from"] = name_parts[0]
                info["language_to"] = name_parts[1]
            else:
                info["language_from"] = "unknown"
        
        return info
    
    def parse_dsl_file(self, dsl_file: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Parse a DSL file and convert it to a dictionary structure
        Returns a tuple of (dictionary_info, dictionary_entries)
        """
        logger.info(f"Parsing DSL file: {dsl_file}")
        
        # Get initial dictionary info from header
        dict_info = self.parse_header(dsl_file)
        dict_name = dict_info["name"]
        
        # Parse the dictionary content
        entries = {}
        current_headword = None
        current_definition = []
        entry_count = 0
        
        # DSL files are typically in UTF-16-LE encoding, but let's try to detect it
        encodings_to_try = ['utf-16-le', 'utf-8', 'windows-1251', 'latin-1']
        
        for encoding in encodings_to_try:
            try:
                with open(dsl_file, 'r', encoding=encoding) as f:
                    # Try to read a bit of the file to test encoding
                    f.read(100)
                    
                    # If we got here, encoding is likely correct
                    logger.info(f"Using {encoding} encoding for {dsl_file}")
                    
                    # Reopen the file with the detected encoding
                    with open(dsl_file, 'r', encoding=encoding) as f:
                        # Skip BOM if present
                        first_char = f.read(1)
                        if first_char != '\ufeff':
                            f.seek(0)
                        
                        in_header = True
                        header_lines = []
                        
                        for line in f:
                            line = line.rstrip()
                            
                            # Skip empty lines
                            if not line:
                                continue
                            
                            # Check if we're still in the header
                            if in_header:
                                if line.startswith('#'):
                                    # Collect header lines for potential description
                                    header_lines.append(line)
                                    
                                    # Process header directives
                                    if line.startswith("#NAME"):
                                        # Extract the display name from the #NAME directive
                                        name_value = line[5:].strip().strip('"')
                                        dict_info["display_name"] = name_value
                                        logger.info(f"Found #NAME directive in DSL file: {name_value}")
                                    elif line.startswith("#INDEX_LANGUAGE"):
                                        language_from = line[16:].strip().strip('"')
                                        dict_info["language_from"] = language_from
                                    elif line.startswith("#CONTENTS_LANGUAGE"):
                                        language_to = line[19:].strip().strip('"')
                                        dict_info["language_to"] = language_to
                                    elif line.startswith("#DESCRIPTION"):
                                        description = line[13:].strip().strip('"')
                                        if not dict_info["description"]:  # Only set if not already set from .ann file
                                            dict_info["description"] = description
                                    continue
                                else:
                                    # If we get here, we're out of the header section
                                    in_header = False
                                    
                                    # If no description is set yet, use the header information
                                    if not dict_info["description"] and header_lines:
                                        # Filter out directive lines and get meaningful content
                                        non_directive_lines = [l for l in header_lines if not any(
                                            l.startswith(d) for d in 
                                            ["#NAME", "#INDEX_LANGUAGE", "#CONTENTS_LANGUAGE", 
                                             "#DESCRIPTION", "#ICON_FILE"])]
                                        
                                        if non_directive_lines:
                                            # Use remaining header content as description
                                            dict_info["description"] = "\n".join(
                                                l.strip('#').strip() for l in non_directive_lines[:5]
                                            )
                            
                            # If line is not indented, it's a headword
                            if not line.startswith('\t'):
                                # Save the previous entry if exists
                                if current_headword is not None:
                                    entries[current_headword] = '\n'.join(current_definition)
                                    entry_count += 1
                                
                                # Start a new entry
                                current_headword = line
                                current_definition = []
                            else:
                                # Add to the current definition (remove the tab)
                                current_definition.append(line[1:])
                    
                    # Add the last entry
                    if current_headword is not None:
                        entries[current_headword] = '\n'.join(current_definition)
                        entry_count += 1
                    
                    # Update entry count in dictionary info
                    dict_info["entry_count"] = entry_count
                    
                    logger.info(f"Parsed {entry_count} entries from {dict_name}")
                    return dict_info, entries
                    
            except UnicodeDecodeError:
                # Try next encoding
                continue
            except Exception as e:
                logger.error(f"Error parsing DSL file {dsl_file} with {encoding} encoding: {e}")
                continue
        
        # If we get here, none of the encodings worked
        logger.error(f"Failed to determine encoding for DSL file {dsl_file}")
        return dict_info, {}
    
    def parse_all_dictionaries(self) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        """
        Parse all DSL files in the dictionary directory
        Returns a tuple of (dictionary_info_dict, dictionaries_dict)
        """
        dsl_files = self.find_dsl_files()
        logger.info(f"Found {len(dsl_files)} DSL files")
        
        dictionary_info = {}
        dictionaries = {}
        
        for dsl_file in dsl_files:
            dict_info, dict_entries = self.parse_dsl_file(dsl_file)
            dict_name = dict_info["name"]
            
            dictionary_info[dict_name] = dict_info
            dictionaries[dict_name] = dict_entries
        
        self.dictionary_info = dictionary_info
        self.dictionaries = dictionaries
        
        return dictionary_info, dictionaries
    
    def save_to_json(self, output_dir: Path) -> None:
        """Save parsed dictionaries to JSON files for faster loading"""
        output_dir.mkdir(exist_ok=True)
        
        # Save dictionary info
        with open(output_dir / "dictionary_info.json", 'w', encoding='utf-8') as f:
            json.dump(self.dictionary_info, f, ensure_ascii=False, indent=2)
        
        # Save each dictionary to a separate file
        for dict_name, entries in self.dictionaries.items():
            dict_file = output_dir / f"{dict_name}.json"
            with open(dict_file, 'w', encoding='utf-8') as f:
                json.dump(entries, f, ensure_ascii=False)
            
            logger.info(f"Saved dictionary {dict_name} to {dict_file}")
    
    @staticmethod
    def load_from_json(json_dir: Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        """Load dictionaries from JSON files"""
        dictionary_info = {}
        dictionaries = {}
        
        info_file = json_dir / "dictionary_info.json"
        if info_file.exists():
            with open(info_file, 'r', encoding='utf-8') as f:
                dictionary_info = json.load(f)
        
        # Load each dictionary
        for dict_file in json_dir.glob("*.json"):
            if dict_file.name == "dictionary_info.json":
                continue
                
            dict_name = dict_file.stem
            with open(dict_file, 'r', encoding='utf-8') as f:
                dictionaries[dict_name] = json.load(f)
            
            logger.info(f"Loaded dictionary {dict_name} from {dict_file}")
        
        return dictionary_info, dictionaries


def clean_dsl_markup(text: str) -> str:
    """
    Remove or convert DSL markup tags to make the text more readable.
    
    Common DSL tags include:
    - [p] - Paragraph
    - [i] - Italic text
    - [b] - Bold text 
    - [u] - Underlined text
    - [c] - Color/formatting control
    - [trn] - Translation (often numbered like [trn1])
    - [com] - Comment (typically grammatical information)
    - [m] - Meaning (often numbered like [m1])
    - [ex] - Example usage
    - [ref] - Reference/link to another dictionary entry
    - [url] - External URL
    
    This function converts formatting tags to Markdown syntax
    and removes most specialized dictionary markup tags.
    """
    # Remove color tags
    text = re.sub(r'\[c\s[^\]]*\](.*?)\[/c\]', r'\1', text)
    
    # Convert basic formatting
    text = re.sub(r'\[b\](.*?)\[/b\]', r'**\1**', text)  # bold
    text = re.sub(r'\[i\](.*?)\[/i\]', r'*\1*', text)    # italic
    text = re.sub(r'\[u\](.*?)\[/u\]', r'_\1_', text)    # underline
    
    # Handle specific semantic tags
    text = re.sub(r'\[trn\d*\](.*?)\[/trn\d*\]', r'\1', text)  # translations
    text = re.sub(r'\[ex\](.*?)\[/ex\]', r'Example: \1', text)  # examples
    text = re.sub(r'\[com\](.*?)\[/com\]', r'(\1)', text)  # comments
    text = re.sub(r'\[p\](.*?)\[/p\]', r'\1', text)  # paragraphs
    text = re.sub(r'\[m\d*\](.*?)\[/m\d*\]', r'\1', text)  # meanings
    
    # Handle references
    text = re.sub(r'\[ref\](.*?)\[/ref\]', r'See: \1', text)
    
    # Remove any remaining tags
    text = re.sub(r'\[[^\]]*\]', '', text)
    text = re.sub(r'\[/[^\]]*\]', '', text)
    
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def convert_dsl_to_html(text: str) -> str:
    """
    Convert DSL markup tags to HTML for better readability.
    
    This function transforms DSL markup tags into their HTML equivalents,
    making the output more readable in browsers and JSON viewers while
    preserving the semantic structure of the dictionary entries.
    
    Common DSL tags include:
    - [p] - Paragraph (converted to <p>)
    - [i] - Italic text (converted to <i>)
    - [b] - Bold text (converted to <b>)
    - [u] - Underlined text (converted to <u>)
    - [c] - Color/formatting control (converted to <span> with styling)
    - [trn] - Translation (converted to <span class="translation">)
    - [com] - Comment/grammatical info (converted to <span class="comment">)
    - [m] - Meaning (converted to <div class="meaning">)
    - [ex] - Example (converted to <div class="example">)
    - [ref] - Reference (converted to <a>)
    """
    # Convert paragraphs
    text = re.sub(r'\[p\](.*?)\[/p\]', r'<p>\1</p>', text)
    
    # Convert basic formatting
    text = re.sub(r'\[b\](.*?)\[/b\]', r'<b>\1</b>', text)
    text = re.sub(r'\[i\](.*?)\[/i\]', r'<i>\1</i>', text)
    text = re.sub(r'\[u\](.*?)\[/u\]', r'<u>\1</u>', text)
    
    # Handle colors and formatting
    text = re.sub(r'\[c\s?[^\]]*\](.*?)\[/c\]', r'<span class="formatted">\1</span>', text)
    
    # Handle specific semantic tags
    text = re.sub(r'\[trn(\d*)\](.*?)\[/trn\1\]', r'<span class="translation" data-num="\1">\2</span>', text)
    text = re.sub(r'\[ex\](.*?)\[/ex\]', r'<div class="example">\1</div>', text)
    text = re.sub(r'\[com\](.*?)\[/com\]', r'<span class="comment">\1</span>', text)
    text = re.sub(r'\[m(\d*)\](.*?)\[/m\1\]', r'<div class="meaning" data-num="\1">\2</div>', text)
    
    # Handle references - use a custom function to properly URL encode references
    def encode_ref(match):
        ref_text = match.group(1)
        # Create a valid URL by encoding the reference text
        # Use data-word attribute to store the original reference text
        safe_href = ref_text.replace(' ', '_').replace('#', '').replace('&', '').replace('?', '')
        return f'<a href="#lookup/{safe_href}" data-word="{ref_text}" class="dictionary-ref">{ref_text}</a>'
    
    text = re.sub(r'\[ref\](.*?)\[/ref\]', encode_ref, text)
    
    # Handle external URLs
    text = re.sub(r'\[url\](.*?)\[/url\]', r'<a href="\1" target="_blank" rel="noopener noreferrer">\1</a>', text)
    
    # Clean up any remaining unclosed/unmatched tags
    text = re.sub(r'\[[^\]]*\]', '', text)
    text = re.sub(r'\[/[^\]]*\]', '', text)
    
    # Convert multiple newlines to proper paragraphs
    text = re.sub(r'\n{2,}', '</p><p>', text)
    
    # Add paragraph tags if they're not already present
    if not text.startswith('<p>'):
        text = '<p>' + text
    if not text.endswith('</p>'):
        text = text + '</p>'
    
    return text


def normalize_headword(headword: str) -> str:
    """Normalize a headword for consistent lookup"""
    # Remove accent marks, diacritics, etc. if needed
    
    # Convert to lowercase
    headword = headword.lower()
    
    # Remove any trailing comments or variants in parentheses
    headword = re.sub(r'\s*\([^)]*\)$', '', headword)
    
    return headword.strip() 