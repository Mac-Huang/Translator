Title: Microsoft Speech Language Translation Corpus
Release: September 18, 2017

Overview
========

This release contains conversational, bilingual speech test and tuning data
for English, Chinese and Japanese collected by Microsoft Research. The package
includes audio data, transcripts, and translations and allows end-to-end
testing of spoken language translation systems on real-world data.

Disclaimer
==========

All data contained in this release has been created using a non-public version
of Skype Translator. NO PRIVATE USER DATA HAS BEEN COLLECTED OR RELEASED.
Instead we hired consultants to have loosely constrained conversations, giving
them a list of predefined topics to talk about and a few related questions
to start the conversations. Topical constraints were loosely enforced so as
to ensure free-form conversations. See the MT Summit paper for more details.

License
=======

See LICENSE.txt or LICENSE.docx for the license terms for this release.

Citation
========

Inside the Paper folder you can find the corpus description paper released
at MT Summit 2017. Please cite this paper when using the MSLT corpus in your
research. A BibTex file is available in the same folder.

Data
====

We release two sets, one containing Test data, the second containing Dev data.
Each set contains data for three languages: English, Chinese and Japanese.
For every utterance, we include the audio file in WAVE format, the disfluent
transcript, a cleaned up, segmented and fluent version of the transcript, and
the translation from English into Chinese or Japanese or vice versa.

- MSLT_Test_EN_20170914

  Contains 10,635 files in total, corresponding to 2,127 utterances.
  For each utterance, there is a German source audio file, two
  German transcript files, and one translation file:
  
  1) *.T0.en.wav contains source audio signal
  2) *.T1.en.snt contains "disfluent, verbatim" human transcripts
  3) *.T2.en.snt contains "fluent, segmented" human transcripts
  4) *.T3.ja.snt contains translation from T2 into Japanese
  5) *.T3.zh.snt contains translation from T2 into Chinese

- MSLT_Test_JA_20170914

  Contains 16,640 files in total, corresponding to 4,160 utterances.
  Same set of files as for the English subset but contains target
  translations from Japanese into English.

- MSLT_Test_ZH_20170914

  Contains 5,140 files in total, corresponding to 1,285 utterances.
  Same set of files as for the English subset but contains target
  translations from Chinese into English.

- MSLT_Dev_EN_20170914

  Contains 11,115 files in total, correspondings to 2,223 utterances.
  Same set of files as for the English Test set.

- MSLT_Dev_JA_20170914

  Contains 12,716 files in total, correspondings to 3,179 utterances.
  Same set of files as for the English Test set but contains target
  translations from from Japanese into English.

- MSLT_Dev_ZH_20170914

  Contains 5,024 files in total, correspondings to 1,256 utterances.
  Same set of files as for the English Test set but contains target
  translations from from Chinese into English.
 
Contact
=======

Send questions or feedback related to this release to <chrife@microsoft.com>.
