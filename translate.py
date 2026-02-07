import os
import polib
from deep_translator import GoogleTranslator
import sys

def translate_po_file(lang_code):
    """
    Automatically translates a .po file for a given language code using deep-translator.
    """
    try:
        # Find the .po file
        po_file_path = os.path.join('locale', lang_code, 'LC_MESSAGES', 'django.po')

        if not os.path.exists(po_file_path):
            print(f"Error: Could not find .po file at {po_file_path}")
            return

        print(f"Loading .po file from {po_file_path}")
        po = polib.pofile(po_file_path)
        
        untranslated_entries = [e for e in po if not e.msgstr and not e.obsolete]
        
        if not untranslated_entries:
            print("No untranslated entries found. Nothing to do.")
            return

        print(f"Found {len(untranslated_entries)} untranslated entries. Translating...")

        for entry in untranslated_entries:
            try:
                # Skip empty msgid
                if not entry.msgid.strip():
                    continue
                
                # The source language is always English in this case
                translated_text = GoogleTranslator(source='auto', target=lang_code).translate(entry.msgid)
                entry.msgstr = translated_text
                print(f"Translated '{entry.msgid}' to '{entry.msgstr}'")
            except Exception as e:
                print(f"Could not translate '{entry.msgid}': {e}")

        print("Saving the translated .po file...")
        po.save()
        print("Done.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python translate.py <language_code>")
        sys.exit(1)
    
    lang_code_to_translate = sys.argv[1]
    translate_po_file(lang_code_to_translate)