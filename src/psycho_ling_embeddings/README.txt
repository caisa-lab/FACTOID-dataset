How to run BFM script on a single text:

from personality.loading_personality import Lexical_personality
personality_path = "./personality"
pers = Lexical_personality(path=personality_path)
loop = tqdm(text, disable=True)
personality_features = Parallel(n_jobs=8, backend="multiprocessing", prefer="processes")(delayed(pers.one_vector_personality)(sentence) for sentence in loop)

--------------------------------------------------------------------------------------
How to run LIWC script:

from LIWC.loading_LIWC import Lexical_LIWC
LIWC_path = "./LIWC"
lex = Lexical_LIWC(path=LIWC_path)
loop = tqdm(text_column, disable=True)
LIWC_features = Parallel(n_jobs=8, backend="multiprocessing", prefer="processes")(delayed(lex.one_vector_LIWC)(sentence) for sentence in loop)