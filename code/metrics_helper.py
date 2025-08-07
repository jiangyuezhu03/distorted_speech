from jiwer import process_characters

result = process_characters(ref, hyp)

subs = result.substitutions
dels = result.deletions
ins = result.insertions
hits = result.hits

mer = (subs + dels + ins) / (subs + dels + ins + hits)
