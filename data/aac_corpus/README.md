# AAC phrase corpora

## vertanen_kristensson_2011/
Crowdsourced AAC-like communications, about 6,100 messages written by Mechanical Turk workers as if they
were using an AAC device. Files are unmodified copies of `sent_{train,dev,test}_aac.txt` and `readme.txt`.

- Source: https://www.aactext.org/imagine/
- Paper: Keith Vertanen and Per Ola Kristensson. *The Imagination of Crowds: Conversational AAC Language
  Modeling using Crowdsourcing and Large Data Sources.* EMNLP 2011, pp. 700–711.
- License: CC BY 4.0.

**How it enters the phrase database.** Run `python retrieval.py import-corpus <files> --source vertanen2011`.
Each message is tagged with the intent of its nearest hand-written seed phrase, and kept only if the cosine
is ≥ 0.6 and beats every other intent by ≥ 0.05. Case- and punctuation-duplicates are dropped.

Of 3,883 unique candidates, 162 matched. A hand review (2026-10-24) dropped 18:
- partner-voice lines ("What do you want to eat?")
- negations ("I am not tired yet.")
- ambiguous lines ("Please.", "I want you.")

It also retagged "That's not right." to No. **Result: 144 imported rows.**

## Not imported
- **COMM2** (aactext.org/comm2): 1,506 messages. Its page says CC BY 4.0, but its readme says CC BY-ND 3.0.
  Unresolved.
- **Vertanen 2017 AAC dialogue corpus:** no license stated.
