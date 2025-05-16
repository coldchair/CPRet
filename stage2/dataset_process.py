from tqdm import tqdm

def filted_by_length(data, max_length=1024):
    """
    Filter out solutions in each sample whose token length exceeds max_length.

    Args:
        data (list): A list of data samples, each containing 'solutions' and 'solutions_length'.
        max_length (int): Maximum allowed token length for a solution.

    Returns:
        list: A filtered list of samples, each retaining only solutions with acceptable length.
    """
    n = len(data)
    new_data = []
    for i in range(n):
        m = len(data[i]['solutions'])

        # Find indices of solutions within the length constraint
        index = [j for j in range(m) if data[i]['solutions_length'][j] <= max_length]

        # Keep only the filtered solutions and lengths
        data[i]['solutions'] = [data[i]['solutions'][j] for j in index]
        data[i]['solutions_length'] = [data[i]['solutions_length'][j] for j in index]

        if len(index) > 0:
            new_data.append(data[i])

    return new_data

# Filter out solutions whose token count exceeds max_length
def process_dataset(dataset, max_length=1024):
    """
    Process a HuggingFace dataset dict by filtering out solutions that exceed a length threshold.

    Args:
        dataset (DatasetDict): HuggingFace dataset containing 'train' and 'test' splits.
        max_length (int): Maximum token length allowed for each solution.

    Returns:
        Tuple: Filtered train and test data as lists.
    """
    traindata = dataset['train'].to_list()
    testdata = dataset['test'].to_list()

    traindata = filted_by_length(traindata, max_length)
    testdata = filted_by_length(testdata, max_length)

    print(f"Train data size: {len(traindata)}")
    print(f"Test data size: {len(testdata)}")

    return traindata, testdata

def process_eval_dataset(queries, corpus, qrels):
    queries = dict(zip([str(x['_id']) for x in queries], [x['text'] for x in queries]))  # Our queries (qid => question)
    corpus = dict(zip([str(x['_id']) for x in corpus], [x['text'] for x in corpus]))  # Our corpus (cid => document)
    relevant_docs = {}  # Query ID to relevant documents (qid => set([relevant_cids])
    for qrel in qrels:
        qid, corpus_ids = qrel["query-id"], qrel["corpus-id"]
        qid = str(qid)
        corpus_ids = str(corpus_ids)
        if qid not in relevant_docs:
            relevant_docs[qid] = set()
        relevant_docs[qid].add(corpus_ids)
    return queries, corpus, relevant_docs