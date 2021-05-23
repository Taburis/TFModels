
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from config import ParaSet as pset
#from module import Transfomer as xtr
import unicodedata
import six
import re

def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")

def whitespace_tokenize(text):
    """ split text into word sequence """
    text = text.strip()
    if not text : return []
    else: return text.split()

def is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
        (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False

class tokenizer(object):
    def __init__(self, is_cased = False):
        self.is_cased = is_cased

    def basic_tokenize(self, text):
        text = convert_to_unicode(text)
        text = whitespace_tokenize(text)
        parsed = []
        for wd in text: 
            if not self.is_cased: wd=wd.lower()
            wd = self.strip_accent(wd)
            ll = re.findall(r"[\w]+|[^\s\w]", wd)
            parsed.extend(ll)
        return parsed

    def strip_accent(self, text):
        text = unicodedata.normalize("NFD",text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn": continue
            output.append(char)
        return "".join(output)
    
        
       
