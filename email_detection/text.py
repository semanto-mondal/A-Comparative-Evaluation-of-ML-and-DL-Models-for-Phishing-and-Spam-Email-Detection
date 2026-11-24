"""Text normalisation.

harmonise()        applied to every email first; removes formatting differences
                   between IWSPA and Enron so models cannot use them as shortcuts.
clean_classical()  input for TF-IDF models (and, without stopword removal and
                   lemmatisation, for the BiLSTM and Transformer).
clean_bert()       input for BERT: the harmonised text, nothing removed.
clean_legacy()     the cleaning used in our first submission; only for the ablation.
dedup_key()        normalised key used to find exact duplicates.
"""

import html
import re

import nltk


def ensure_nltk_data():
    for package in ["stopwords", "wordnet", "omw-1.4", "vader_lexicon"]:
        try:
            nltk.download(package, quiet=True)
        except Exception as err:  # offline machines: data may already be installed
            print(f"Could not download NLTK package '{package}': {err}")


ensure_nltk_data()

from nltk.corpus import stopwords          # noqa: E402  (needs the data above)
from nltk.stem import WordNetLemmatizer    # noqa: E402

NEGATIONS = {"not", "no", "nor", "never"}
STOPWORDS = set(stopwords.words("english")) - NEGATIONS
LEGACY_STOPWORDS = set(stopwords.words("english")) - {"not"}
LEMMATIZER = WordNetLemmatizer()

URL_TOKEN = " [URL] "
EMAIL_TOKEN = " [EMAIL] "
NUM_TOKEN = " [NUM] "

# Encoding and markup artefacts
RE_IWSPA_LINK = re.compile(r"<+\s*link\s*>+", re.I)                  # IWSPA anonymised links: <<link>>
RE_QP_SOFTBREAK = re.compile(r"=\r?\n")                               # quoted-printable soft line break
RE_QP_CODES = re.compile(r"=(20|3D|09|0A|0D)", re.I)                  # quoted-printable codes
RE_QP_JOINED = re.compile(r"(?<=\w)= (?=\w)")                         # already-joined break: "Ite= m"
RE_QP_LONE = re.compile(r"(?<=\s)=(?=\s)")
RE_HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
RE_HTML_TAG = re.compile(r"<[^>]{1,500}>")
RE_CSS_RULE = re.compile(r"[.#]?[\w-]+(?:\s*,\s*[.#]?[\w-]+)*\s*\{[^{}]{0,400}:[^{}]{0,400}\}")
RE_CSS_DECL = re.compile(r"\b(?:font|color|margin|padding|border|text|line|background|width|height|display|"
                         r"letter|vertical|mso)[\w-]*\s*:\s*[^;\n]{1,80};", re.I)

# Enron is pre-tokenised ("http : / / www . x . com", "it ' s"); these undo that
RE_SPACED_SCHEME = re.compile(r"\b(https?|ftp)\s*:\s*/\s*/\s*", re.I)
RE_SPACED_WWW = re.compile(r"\bwww\s*\.\s*", re.I)
RE_SPACED_DOT = re.compile(r"(?<=[a-z0-9])\s+\.\s+"
                           r"(?=(?:www|com|net|org|edu|gov|biz|info|co|uk|de|us|html?|asp|php|cgi)\b)", re.I)
RE_SPACED_URL_SLASH = re.compile(r"(?<=://)(\S+)\s+/\s+(?=\S)")
RE_SPACED_APOS = re.compile(r"\s+'\s*(s|t|re|ve|ll|d|m)\b", re.I)
RE_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,!?;:%)\]])")
RE_SPACE_AFTER_OPEN = re.compile(r"([(\[$])\s+")
RE_DASH_RUNS = re.compile(r"(?:-\s*){3,}")
RE_UNDERSCORE_RUNS = re.compile(r"_{3,}")

# Links and addresses
RE_URL = re.compile(r"(?:https?|ftp)://\S+|\bwww\.\S+|\b[\w.-]+\.(?:com|net|org|biz|info|edu|gov)(?:/\S*)?\b", re.I)
RE_EMAIL = re.compile(r"\b[\w.+-]+\s*@\s*[\w-]+(?:\s*\.\s*[\w-]+)+\b")

RE_WS = re.compile(r"\s+")
RE_NUM = re.compile(r"\b\d[\d,.:/-]*\b")
RE_NON_ALPHA = re.compile(r"[^a-z\[\]\s]")
PLACEHOLDER_WORDS = {"[url]": " urltoken ", "[email]": " emailtoken ", "[num]": " numtoken "}

QP_CODES = {"20": " ", "3D": "=", "09": " ", "0A": " ", "0D": " "}


def harmonise(text: str) -> str:
    """Bring IWSPA and Enron emails to the same format.

    URLs and e-mail addresses are replaced by [URL] and [EMAIL] instead of
    being deleted, so the presence of a link is kept as a signal.
    """
    t = html.unescape(str(text))

    t = RE_QP_SOFTBREAK.sub("", t)
    t = RE_QP_CODES.sub(lambda m: QP_CODES[m.group(1).upper()], t)
    t = RE_QP_JOINED.sub("", t)
    t = RE_QP_LONE.sub(" ", t)

    t = RE_IWSPA_LINK.sub(URL_TOKEN, t)
    t = RE_HTML_COMMENT.sub(" ", t)
    t = RE_HTML_TAG.sub(" ", t)
    t = RE_CSS_RULE.sub(" ", t)
    t = RE_CSS_DECL.sub(" ", t)

    # Repair Enron's spaced-out URLs before detecting them
    t = RE_SPACED_SCHEME.sub(lambda m: m.group(1).lower() + "://", t)
    t = RE_SPACED_WWW.sub("www.", t)
    t = RE_SPACED_DOT.sub(".", t)
    t = RE_SPACED_URL_SLASH.sub(r"\1/", t)

    t = RE_EMAIL.sub(EMAIL_TOKEN, t)
    t = RE_URL.sub(URL_TOKEN, t)

    # Repair Enron's spaced-out punctuation
    t = RE_SPACED_APOS.sub(lambda m: "'" + m.group(1), t)
    t = RE_SPACE_BEFORE_PUNCT.sub(r"\1", t)
    t = RE_SPACE_AFTER_OPEN.sub(r"\1", t)
    t = RE_DASH_RUNS.sub(" --- ", t)
    t = RE_UNDERSCORE_RUNS.sub(" ___ ", t)

    return RE_WS.sub(" ", t).strip()


def clean_classical(text: str, remove_stopwords: bool = True, lemmatize: bool = True) -> str:
    """Lowercase, numbers -> [NUM], drop punctuation, optional stopword removal and lemmatisation."""
    t = text.lower()
    t = RE_NUM.sub(NUM_TOKEN, t)
    for placeholder, word in PLACEHOLDER_WORDS.items():
        t = t.replace(placeholder, word)
    t = RE_NON_ALPHA.sub(" ", t).replace("[", " ").replace("]", " ")

    tokens = [w for w in t.split() if len(w) > 1]
    if remove_stopwords:
        tokens = [w for w in tokens if w not in STOPWORDS]
    if lemmatize:
        tokens = [LEMMATIZER.lemmatize(w) for w in tokens]
    return " ".join(tokens)


def clean_sequence(text: str) -> str:
    """Input for BiLSTM/Transformer: like clean_classical, but word order is kept intact."""
    return clean_classical(text, remove_stopwords=False, lemmatize=False)


def clean_bert(text: str) -> str:
    """BERT gets the harmonised text as is; its tokenizer handles case and punctuation."""
    return text


def clean_legacy(text: str) -> str:
    """Cleaning from our first submission (URLs, numbers and HTML deleted). Ablation only."""
    text = str(text).lower()
    text = " ".join(w for w in text.split() if w not in LEGACY_STOPWORDS)
    text = re.sub(r"\W", " ", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\s+[a-zA-Z]\s+", " ", text)
    text = re.sub(r"\^[a-zA-Z]\s+", " ", text)
    text = re.sub(r"\s+", " ", text, flags=re.I)
    return " ".join(LEMMATIZER.lemmatize(w) for w in text.split())


def dedup_key(text: str) -> str:
    """Lowercase alphanumerics only: two emails with the same key are exact duplicates."""
    return RE_WS.sub(" ", re.sub(r"[^a-z0-9]", " ", text.lower())).strip()
