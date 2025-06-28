"""
This code was architected, developed, and programmed by Ben-Hur Varriano for Sapiens Technology®️ (as well as all other AI algorithms of the company),
and any unauthorized alteration, adaptation, and/or distribution, as well as public comments and/or postings regarding the operation and/or mathematics involved in the algorithm, are strictly prohibited.
Failure to comply with these rules may result in legal action against the author by our team of attorneys.

This code is an extension of the SCNet algorithm with improvements in training and inference that allow it to run on custom devices.
It features an infinite context window that makes predictions through semantic comparisons between the user prompt and the inputs of the training or adjustment samples.
The SCN network operates on a layer of the HurNetTorch network that calculates weights in a single step without backpropagation in the fine-tuning training, which allows for huge speed gains during this phase.
The data reading and processing functions work with iterative streams, which allows the use of huge data sets without memory overflow or performance loss.

We named the network SCN, an abbreviation of "Semantic Comparison Network", referring to the underlying algorithm SCNet (also authored by Ben-Hur Varriano) that originated the current code.
"""
# --------------------------> A SAPIENS TECHNOLOGY®️ PRODUCTION) <--------------------------
class SCN():
    def __init__(self, device=None, user_id=0, parallel_prediction=False, minimum_probability_for_candidates=0.9, show_errors=True):
        try:
            from warnings import filterwarnings
            filterwarnings('ignore')
            self.__local_device = device.lower().strip() if type(device) == str else None
            self.user_id = int(user_id) if type(user_id) in (bool, int, float) else 0
            self.__parallel_prediction = bool(parallel_prediction) if type(parallel_prediction) in (bool, int, float) else False
            self.minimum_probability_for_candidates = min((1.0, max((0.0, float(minimum_probability_for_candidates))))) if type(minimum_probability_for_candidates) in (bool, int, float) else 0.9
            self.__show_errors = bool(show_errors) if type(show_errors) in (bool, int, float) else True
            from os.path import split, exists, splitext, basename, join, getsize, dirname, isdir, isfile
            from scnetwork import SCNet
            from os import remove, sysconf, makedirs, replace, stat, fsync, listdir
            from pathlib import Path
            from psutil import virtual_memory
            from tqdm import tqdm
            from functools import partialmethod
            from shutil import get_terminal_size, copy2, copyfile
            from urllib.request import urlopen
            from json import load as json_load
            from io import TextIOWrapper
            from csv import reader as csv_reader
            from mmap import mmap as MemoryMap, ACCESS_READ
            from tempfile import gettempdir, NamedTemporaryFile
            from unicodedata import normalize, category
            from numpy import fromstring, mean
            from time import sleep
            from errno import EXDEV
            from sys import platform
            from ctypes import CDLL, c_uint64
            from re import sub, DOTALL
            from torch import save, load, cuda, float32
            from hurnet_torch import HurNetTorch
            from glob import glob
            from urllib.parse import urlparse
            self.__split = split
            self.__SCNet = SCNet(
                device=device,
                user_id=user_id,
                parallel_prediction=parallel_prediction,
                minimum_probability_for_candidates=minimum_probability_for_candidates,
                show_errors=False
            )
            self.__temporary_path = ''
            self.__exists = exists
            self.__remove = remove
            self.__model_path = ''
            self.__Path = Path
            self.__splitext = splitext
            self.__basename = basename
            self.__add_fit = False
            self.__sysconf = sysconf
            self.__virtual_memory = virtual_memory
            self.__tqdm = tqdm
            self.__partialmethod = partialmethod
            self.__get_terminal_size = get_terminal_size
            self.__urlopen = urlopen
            self.__json_load = json_load
            self.__TextIOWrapper = TextIOWrapper
            self.__csv_reader = csv_reader
            self.__MemoryMap = MemoryMap
            self.__ACCESS_READ = ACCESS_READ
            self.__gettempdir = gettempdir
            self.__join = join
            self.__normalize = normalize
            self.__category = category
            self.__fromstring = fromstring
            self.__separators = ('?', '.', ';', '!', '\n')
            self.__sleep = sleep
            self.__getsize = getsize
            self.__block_size = 50
            self.__NamedTemporaryFile = NamedTemporaryFile
            self.parameters_number = 0
            self.__dirname = dirname
            self.__isdir = isdir
            self.__makedirs = makedirs
            self.__replace = replace
            self.__tokenizer = 'gpt-4'
            self.__EXDEV = EXDEV
            self.__stat = stat
            self.__platform = platform
            self.__CDLL = CDLL
            self.__c_uint64 = c_uint64
            self.__fsync = fsync
            self.__inputs = []
            self.__outputs = []
            self.__fine_tuning = False
            self.__fit_model_path = ''
            self.__hurnet_architecture = False
            self.__hur_net = None
            self.__trained = False
            self.__loaded = False
            self.__candidates = []
            self.fit_probability = 0.0
            self.probability = 0.0
            self.__sub = sub
            self.__DOTALL = DOTALL
            self.__hidden_layers = []
            self.__string = ''
            self.__training_finished = False
            self.__mean = mean
            self.__state_hurnet = {}
            self.__add_hur = False
            self.__progress = True
            self.__save = save
            self.__listdir = listdir
            self.__isfile = isfile
            self.__copy2 = copy2
            self.__path_of_the_loaded_model = ''
            self.__load = load
            self.__load_SCNet = SCNet
            self.__cuda = cuda
            self.__HurNet = HurNetTorch
            self.__float32 = float32
            self.__glob = glob
            self.__copyfile = copyfile
            self.__urlparse = urlparse
        except Exception as error:
            if self.__show_errors: print('ERROR in __init__: '+str(error))
    def __del__(self):
        try:
            if len(self.__temporary_path) > 0 and self.__exists(self.__temporary_path):
                if not self.__temporary_path.endswith('.scn01') and not self.__temporary_path.endswith('.scn02'): self.__remove(self.__temporary_path)
            model_path = str(self.__model_path).strip()
            if len(model_path) > 0:
                def _list_model_files(model_directory_path=''):
                    directory, file_name = self.__split(model_directory_path)
                    model_directory = self.__Path(directory)
                    model_directory = model_directory if model_directory.is_dir() else self.__Path('./')
                    if not model_directory.is_dir(): return []
                    file_name, _ = self.__splitext(self.__basename(model_directory_path))
                    network_model_files = model_directory.glob(f'{file_name}.scn02')
                    fitting_model_files = model_directory.glob(f'{file_name}.scn03') if not self.__add_fit else []
                    return [str(file_path) for file_path in network_model_files] + [str(file_path) for file_path in fitting_model_files]
                list_model_files = _list_model_files(model_directory_path=model_path)
                for model_file in list_model_files: self.__overwriteWithHexadecimal(file_path=model_file, progress=self.__progress)
                self.__model_path = ''
        except Exception as error:
            if self.__show_errors: print('ERROR in __del__: '+str(error))
    def __getBufferSize(self):
        try:
            buffer_size = 1048576
            try: buffer_size = int((self.__sysconf('SC_PAGE_SIZE') * self.__sysconf('SC_AVPHYS_PAGES')) * 0.1)
            except: buffer_size = int(self.__virtual_memory().available * 0.1)
            return buffer_size
        except Exception as error:
            if self.__show_errors: print('ERROR in __getBufferSize: '+str(error))
            return 1048576
    def __setTqdm(self, disable=True):
        try:
            disable = bool(disable) if type(disable) in (bool, int, float) else True
            self.__tqdm.__init__ = self.__partialmethod(self.__tqdm.__init__, disable=disable)
            return True
        except Exception as error:
            if self.__show_errors: print('ERROR in __setTqdm: '+str(error))
            return False
    def __updateTqdm(self, total=1, description=''):
        try:
            total = int(total) if type(total) in (int, float) else 1
            description = description.strip() if type(description) == str else str(description).strip()
            progress_bar = self.__tqdm(total=total, desc=description, position=0, ncols=self.__get_terminal_size().columns, leave=True)
            return progress_bar
        except Exception as error:
            if self.__show_errors: print('ERROR in __updateTqdm: '+str(error))
            return False    
    def __splitText(self, dataset_path='', end_tag=''):
        try:
            lower_path = dataset_path.lower()
            remote_flag = lower_path.startswith(('http://', 'https://'))
            if lower_path.endswith('.json'):
                data_stream = self.__urlopen(dataset_path) if remote_flag else open(dataset_path, 'rb')
                try:
                    parsed_data = self.__json_load(data_stream)
                    if isinstance(parsed_data, dict):
                        if 'data' in parsed_data and isinstance(parsed_data['data'], list): data_items = parsed_data['data']
                        else:
                            data_items = []
                            for value in parsed_data.values():
                                if isinstance(value, list):
                                    data_items = value
                                    break
                    elif isinstance(parsed_data, list): data_items = parsed_data
                    else: data_items = []
                    for entry in data_items:
                        if not isinstance(entry, dict): continue
                        input_text = ''
                        for field in ('input', 'prompt', 'question'):
                            if field in entry and isinstance(entry[field], str):
                                input_text = entry[field]
                                break
                        if not input_text:
                            first_key = next(iter(entry), None)
                            input_text = str(entry.get(first_key, '')) if first_key else ''
                        output_text = ''
                        for field in ('output', 'answer', 'response'):
                            if field in entry and isinstance(entry[field], str):
                                output_text = entry[field]
                                break
                        if not output_text:
                            last_key = None
                            for key in entry: last_key = key
                            output_text = str(entry.get(last_key, '')) if last_key else ''
                        input_text, output_text = input_text.strip(), output_text.strip()
                        if input_text == output_text: output_text = ''
                        yield str(f'{input_text}\n{output_text}').strip()
                finally: data_stream.close()
                return
            if lower_path.endswith('.csv'):
                if remote_flag:
                    remote_stream = self.__urlopen(dataset_path)
                    text_stream = self.__TextIOWrapper(remote_stream, encoding='utf-8')
                    csv_reader_object = self.__csv_reader(text_stream)
                else:
                    file_stream = open(dataset_path, 'r', encoding='utf-8', newline='')
                    csv_reader_object = self.__csv_reader(file_stream)
                try:
                    header_row = next(csv_reader_object, None)
                    if header_row:
                        input_index = next((header_row.index(name) for name in ('input', 'prompt', 'question') if name in header_row), 0)
                        output_index = next((header_row.index(name) for name in ('output', 'answer', 'response') if name in header_row), len(header_row) - 1)
                    else: input_index, output_index = 0, 0
                    for row_data in csv_reader_object:
                        if not row_data: continue
                        input_text = row_data[input_index] if input_index < len(row_data) else ''
                        output_text = row_data[output_index] if output_index < len(row_data) else ''
                        input_text, output_text = input_text.strip(), output_text.strip()
                        if input_text == output_text: output_text = ''
                        yield str(f'{input_text}\n{output_text}').strip()
                finally:
                    if remote_flag: text_stream.detach(), remote_stream.close()
                    else: file_stream.close()
                return
            separator_sequence = end_tag.encode('utf-8')
            sequence_length, buffer_size = len(separator_sequence), self.__getBufferSize()
            if remote_flag:
                remote_stream, buffer_capacity, data_buffer = self.__urlopen(dataset_path), buffer_size, bytearray()
                while True:
                    chunk_bytes = remote_stream.read(buffer_capacity)
                    if not chunk_bytes: break
                    data_buffer.extend(chunk_bytes)
                    start_position = 0
                    while True:
                        position = data_buffer.find(separator_sequence, start_position)
                        if position == -1: break
                        yield str(data_buffer[start_position:position].decode('utf-8')).strip()
                        start_position = position + sequence_length
                    del data_buffer[:start_position]
                if data_buffer: yield str(data_buffer.decode('utf-8')).strip()
                remote_stream.close()
                return
            if self.__getsize(dataset_path) < buffer_size:
                blocks = []
                with open(dataset_path, 'r', encoding='utf-8') as file: blocks = str(file.read()).strip().split(end_tag)
                if len(blocks) > 0:
                    for block in blocks: yield block
                    return
            file_stream = open(dataset_path, 'rb')
            memory_map, start_position = self.__MemoryMap(file_stream.fileno(), 0, access=self.__ACCESS_READ), 0
            while True:
                position = memory_map.find(separator_sequence, start_position)
                if position == -1: break
                yield str(memory_map[start_position:position].decode('utf-8')).strip()
                start_position = position + sequence_length
            if start_position < len(memory_map): yield str(memory_map[start_position:].decode('utf-8')).strip()
            memory_map.close(), file_stream.close()
        except Exception as error:
            if self.__show_errors: print('ERROR in __splitText: '+str(error))
            yield ''
    def __getID(self, user_id=0):
        try:
            temporary_directory = self.__gettempdir()
            filename = self.__join(temporary_directory, str(user_id))
            if not self.__exists(filename): return 0
            with open(filename, 'r') as file_handle: last_id = file_handle.read()
            return int(last_id)
        except Exception as error:
            if self.__show_errors: print('ERROR in __getID: '+str(error))
            return 0
    def __textualComparison(self, text1='', text2='', consider_length=True):
        try:
            probability = 0.0
            text1, text2 = str(text1).strip(), str(text2).strip()
            consider_length = bool(consider_length) if type(consider_length) in (bool, int, float) else True
            def _remove_accents(text=''): return ''.join(character for character in self.__normalize('NFD', text) if self.__category(character) != 'Mn')
            tokens1, tokens2 = text1.split(), text2.split()
            length1, length2 = len(tokens1), len(tokens2)
            search, target = (tokens1, tokens2) if length1 < length2 else (tokens2, tokens1)
            overall_score = 0.0
            for token1 in search:
                characters1 = token1.split()
                maximum_score = 0.0
                for token2 in target:
                    characters2, score, one_third = token2.split(), 0.0, 1.0 / 3.0
                    for char1, char2 in zip(characters1, characters2):
                        if _remove_accents(text=char1).lower() == _remove_accents(text=char2).lower(): score += one_third
                        if _remove_accents(text=char1) == _remove_accents(text=char2): score += one_third
                        if char1.lower() == char2.lower(): score += one_third
                    if score > maximum_score: maximum_score = score
                overall_score += maximum_score
            length_difference = (1 - (len(search) / max((1, len(target))))) / 2
            probability = (overall_score / max((1, len(search))))
            if consider_length and probability > length_difference: probability = probability - length_difference
            return probability
        except Exception as error:
            print('ERROR in __textualComparison: ' + str(error))
            return 0.0
    def __setID(self, user_id=0, last_id=0):
        try:
            temporary_directory = self.__gettempdir()
            filename = self.__join(temporary_directory, str(user_id))
            with open(filename, 'w') as file_handle: file_handle.write(str(last_id))
            return True
        except Exception as error:
            if self.__show_errors: print('ERROR in __setID: '+str(error))
            return False
    def __standardAdjustmentForecast(self, input_prompt=[], vector_matrix=[]):
        try:
            result_dictionary = {'id': 0, 'relationship_id': 0, 'index': 0, 'prompt': '', 'answer': '', 'probability': 0.0}
            if vector_matrix in ((), []): return result_dictionary
            input_text_x = self.__SCNet.embeddingForText(embedding=input_prompt, encoder=self.__tokenizer, strip=True) if type(input_prompt) in (tuple, list) else input_prompt
            maximum_id, maximum_relationship_id = 0, 0
            maximum_probability, maximum_index, old_prompt, maximum_forecast = 0.0, 0, '', ''
            last_id = self.__getID(user_id=self.user_id)
            for index, vector in enumerate(vector_matrix):
                tokens = self.__SCNet.embeddingForText(embedding=vector, encoder=self.__tokenizer, strip=True)
                if '<|end|>' not in tokens: continue
                id, relationship_id, input_text_y, output_text = tokens.split('<|end|>')
                id, relationship_id = int(id), int(relationship_id)
                if relationship_id == 0 or relationship_id == last_id:
                    probability = self.__textualComparison(text1=input_text_x, text2=input_text_y, consider_length=True)
                    if probability > maximum_probability: maximum_id, maximum_relationship_id, maximum_probability, maximum_index, old_prompt, maximum_forecast = id, relationship_id, probability, index, input_text_y, output_text
            result_dictionary = {'id': maximum_id, 'relationship_id': maximum_relationship_id, 'index': maximum_index, 'prompt': old_prompt, 'answer': maximum_forecast, 'probability': maximum_probability}
            self.__setID(user_id=self.user_id, last_id=maximum_id)
            return result_dictionary
        except Exception as error:
            if self.__show_errors: print('ERROR in __standardAdjustmentForecast: '+str(error))
            return {'id': 0, 'relationship_id': 0, 'index': 0, 'prompt': '', 'answer': '', 'probability': 0.0}
    def __iterNumericVectors(self, model_path=''):
        try:
            if len(model_path) < 1: model_path = self.__temporary_path
            if self.__exists(model_path):
                buffering = self.__getBufferSize()
                buffering = min(buffering, 2147483647)
                with open(model_path, 'r', buffering=buffering) as open_file:
                    for line in open_file:
                        embeddings = self.__fromstring(line, dtype=int, sep=' ')
                        if embeddings.size == 0: continue
                        yield embeddings.tolist()
            else: yield []
        except Exception as error:
            if self.__show_errors:
                print('ERROR in __iterNumericVectors: '+str(error))
                print('model_path: '+str(model_path))
            yield []
    def __vectorComparison(self, input_vector=[], vector_matrix=[]):
        try:
            if not self.__parallel_prediction: return self.__SCNet.alternativeVectorComparison(input_vector=input_vector, vector_matrix=vector_matrix)
            if not input_vector or not vector_matrix: return (-1, [])
            return self.__SCNet.vectorComparison(input_vector=input_vector, vector_matrix=vector_matrix)
        except Exception as error:
            if self.__show_errors: print('ERROR in __vectorComparison: '+str(error))
            return (-1, [])
    def __extractResponse(self, prompt='', text=''):
        try:
            prompt, text = str(prompt).strip(), str(text).strip()
            if len(prompt) < 1 or len(text) < 1: return ''
            separators, input_segment = self.__separators, None
            for separator in separators:
                index_position = text.find(separator)
                if index_position != -1 and index_position > 0 and index_position < len(text) - 1:
                    input_segment = text[:index_position + 1]
                    break
            if input_segment is None: return text
            prompt_words = self.__SCNet.normalization(input_text=prompt).split()
            total_words = len(prompt_words)
            if total_words == 0: return text
            lower_case_input, matching_count = self.__SCNet.normalization(input_text=input_segment), 0
            for word in prompt_words:
                if word.lower() in lower_case_input: matching_count += 1
            if matching_count * 2 >= total_words: return text[len(input_segment):].strip()
            return text
        except Exception as error:
            if self.__show_errors: print('ERROR in __extractResponse: '+str(error))
            return text
    def __getStream(self, final_answer='', fit_probability=0.0, probability=0.0, interval=0.0):
        try:
            embeddings = self.__SCNet.textForEmbedding(text=final_answer, length=None, encoder=self.__tokenizer)
            for embedding in embeddings:
                token = self.__SCNet.embeddingForText(embedding=[embedding], encoder=self.__tokenizer, strip=False)
                yield {'answer': token, 'fit_probability': fit_probability, 'probability': probability}
                self.__sleep(interval)
        except Exception as error:
            if self.__show_errors: print('ERROR in __getStream: '+str(error))
            yield {'answer': '', 'fit_probability': fit_probability, 'probability': probability}
    def __processDataset(self, dataset_path='', string='', precision=1.0, tokenizer='gpt-4', context_window=float('inf'), end_tag='', progress=True):
        try:
            self.__setTqdm(disable=not progress)
            progress_bar_1 = self.__updateTqdm(total=6, description='Tokenizing data')
            progress_bar_1.update(1)
            max_sentence_count, new_max_sentence_count = 0, 0
            total_bytes, arbitrary_total, bytes_count = 0, 1000000, 0
            has_dataset_path = len(dataset_path) > 0
            has_string, split_text, parts_number = len(string) > 0, [], 0
            progress_bar_1.update(1)
            if self.__parallel_prediction:
                if context_window < float('inf'): max_sentence_count = context_window
                else:
                    total_bytes = self.__getsize(self.__temporary_path) if self.__exists(self.__temporary_path) else arbitrary_total
                    total_bytes += len(string.encode('utf-8'))
                    if total_bytes < 1: total_bytes = arbitrary_total
                    progress_bar_2 = self.__updateTqdm(total=total_bytes, description='Reading data')
                    if has_dataset_path:
                        for piece in self.__splitText(dataset_path=dataset_path, end_tag=end_tag):
                            if len(piece.strip()) > 0:
                                piece = rf'{piece}'
                                sentence_count = len(piece.split())
                                if sentence_count > max_sentence_count: max_sentence_count = sentence_count
                                current_bytes = len(piece.encode('utf-8'))
                                progress_bar_2.update(current_bytes)
                                bytes_count += current_bytes
                        max_sentence_count = int(round(max_sentence_count + (max_sentence_count/1.5)))
                    if has_string:
                        split_text = string.split(end_tag)
                        for piece in split_text:
                            if len(piece.strip()) > 0:
                                piece = rf'{piece}'
                                sentence_count = len(piece.split())
                                if sentence_count > new_max_sentence_count: new_max_sentence_count = sentence_count
                                current_bytes = len(piece.encode('utf-8'))
                                progress_bar_2.update(current_bytes)
                                bytes_count += current_bytes
                        new_max_sentence_count = int(round(new_max_sentence_count + (new_max_sentence_count/1.5)))
                    max_sentence_count = max((max_sentence_count, new_max_sentence_count))
                    progress_bar_2.n = total_bytes
                    progress_bar_2.refresh()
                    progress_bar_2.close()
                max_sentence_count = max((1, int(round(max_sentence_count*precision))))
                self.__block_size = max_sentence_count
            else:
                progress_bar_2 = self.__updateTqdm(total=1, description='Reading data')
                progress_bar_2.n = 1
                progress_bar_2.refresh()
                progress_bar_2.close()
                max_sentence_count = None
            progress_bar_1.update(1)
            temporary_file = self.__NamedTemporaryFile(delete=False, mode='w', encoding='utf-8', suffix='.temp')
            temporary_path = temporary_file.name
            state = 'a' if self.__exists(temporary_path) else 'w'
            total_bytes, bytes_count = self.__getsize(temporary_path) if self.__exists(temporary_path) else arbitrary_total, 0
            total_bytes += len(string.encode('utf-8'))
            if total_bytes < 1: total_bytes = arbitrary_total
            progress_bar_3, embeddings = self.__updateTqdm(total=total_bytes, description='Embedding data'), [0]
            progress_bar_1.update(1)
            if has_dataset_path:
                with open(temporary_path, state, encoding='utf-8') as output_file:
                    for piece in self.__splitText(dataset_path=dataset_path, end_tag=end_tag):
                        if len(piece.strip()) > 0:
                            piece = rf'{piece}'
                            embeddings = self.__SCNet.textForEmbedding(text=piece, length=max_sentence_count, encoder=tokenizer)
                            output_file.write(' '.join(map(str, embeddings))+'\n')
                            parts_number += 1
                            current_bytes = len(piece.encode('utf-8'))
                            progress_bar_3.update(current_bytes)
                            bytes_count += current_bytes
            state = 'a' if self.__exists(temporary_path) else 'w'
            progress_bar_1.update(1)
            if has_string:
                if len(split_text) < 1: split_text = string.split(end_tag)
                with open(temporary_path, state, encoding='utf-8') as output_file:
                    for piece in split_text:
                        if len(piece.strip()) > 0:
                            piece = rf'{piece}'
                            embeddings = self.__SCNet.textForEmbedding(text=piece, length=max_sentence_count, encoder=tokenizer)
                            output_file.write(' '.join(map(str, embeddings))+'\n')
                            parts_number += 1
                            current_bytes = len(piece.encode('utf-8'))
                            progress_bar_3.update(current_bytes)
                            bytes_count += current_bytes
            progress_bar_3.n = total_bytes if total_bytes > 0 else bytes_count
            progress_bar_3.refresh()
            progress_bar_3.close()
            self.parameters_number = parts_number * (max_sentence_count if self.__parallel_prediction else len(embeddings))
            progress_bar_1.update(1)
            progress_bar_1.close()
            return temporary_path
        except Exception as error:
            if self.__show_errors: print('ERROR in __processDataset: '+str(error))
            return self.__temporary_path
    def __changeExtension(self, model_path='', old_extension='scn02', new_extension='scn01'):
        try:
            if model_path.endswith('.'+old_extension):
                model_path_split = model_path.split('.')
                model_path_split[-1] = new_extension
                model_path = '.'.join(model_path_split)
            return model_path
        except Exception as error:
            if self.__show_errors: print('ERROR in __changeExtension: '+str(error))
            return model_path
    def __format_numbers(self, data_number=0, is_tokens=False):
        try:
            if data_number < 1_000: return data_number if is_tokens else f'{data_number}U'
            elif data_number < 1_000_000: return f'{data_number // 1_000}K'
            elif data_number < 1_000_000_000: return f'{data_number // 1_000_000}M'
            elif data_number < 1_000_000_000_000: return f'{data_number // 1_000_000_000}B'
            else: return f'{data_number // 1_000_000_000_000}T'
        except Exception as error:
            if self.__show_errors: print('ERROR in __format_numbers: '+str(error))
            return data_number
    def __moveAndRenameStream(self, source_path='', destination_path='', formatted_params='', progress=True):
        try:
            self.__setTqdm(disable=not progress)
            progress_bar = self.__updateTqdm(total=6, description=f'Generating {formatted_params} parameters')
            progress_bar.update(1)
            destination_directory = self.__dirname(destination_path)
            if destination_directory and not self.__isdir(destination_directory): self.__makedirs(destination_directory, exist_ok=True)
            try:
                self.__replace(source_path, destination_path)
                progress_bar.update(5)
                return True
            except OSError as error:
                if error.errno != self.__EXDEV: raise
            progress_bar.update(1)
            source_file = open(source_path, 'rb')
            destination_file = open(destination_path, 'wb')
            source_descriptor = source_file.fileno()
            total_size = self.__stat(source_path).st_size
            buffer_size = self.__getBufferSize()
            punch_function, offset = None, 0
            destination_descriptor = destination_file.fileno()
            progress_bar.update(1)
            if self.__platform.startswith('linux'):
                libc = self.__CDLL('libc.so.6', use_errno=True)
                punch_hole_flag, keep_size_flag = 0x02, 0x01
                def perform_punch(offset: int, length: int) -> None: libc.fallocate(source_descriptor, punch_hole_flag | keep_size_flag, self.__c_uint64(offset), self.__c_uint64(length))
                punch_function = perform_punch
            progress_bar.update(1)
            progress_bar_bytes = self.__updateTqdm(total=total_size, description='Processing data')
            while True:
                data_block = source_file.read(buffer_size)
                if not data_block: break
                destination_file.write(data_block)
                destination_file.flush()
                self.__fsync(destination_descriptor)
                if punch_function: punch_function(offset, len(data_block))
                offset += len(data_block)
                progress_bar_bytes.update(len(data_block))
            progress_bar_bytes.close()
            progress_bar.update(1)
            source_file.close()
            destination_file.close()
            self.__remove(source_path)
            self.__temporary_path = destination_path
            progress_bar.update(1)
            progress_bar.close()
            return True
        except Exception as error:
            if self.__show_errors: print('ERROR in __moveAndRenameStream: '+str(error))
            return False
    def __setInputsAndOutputs(self, prompt='', answer=''):
        try:
            if self.__add_hur: prompt = self.__SCNet.normalization(input_text=prompt)
            input_embeddings = self.__SCNet.textForEmbedding(text=str(prompt).strip(), length=self.__block_size, encoder=self.__tokenizer)
            output_embeddings = self.__SCNet.textForEmbedding(text=str(answer).strip(), length=self.__block_size, encoder=self.__tokenizer)
            self.__inputs.append(input_embeddings), self.__outputs.append(output_embeddings)
            return True
        except Exception as error:
            if self.__show_errors: print('ERROR in __setInputsAndOutputs: '+str(error))
            return False
    def __predict(self, prompt='', max_tokens=None, min_fit_probability=0.7, min_probability=0.01, generalization=True, stream=False, interval=0.0):
        try:
            answer_dictionary = {'answer': '', 'fit_probability': 0.0, 'probability': 0.0}
            prompt, output_text = str(prompt).strip(), ''
            max_tokens = int(max_tokens) if type(max_tokens) in (bool, int, float) else None
            min_fit_probability = float(min_fit_probability) if type(min_fit_probability) in (bool, int, float) else 0.7
            min_probability = min((1.0, max((0.0, float(min_probability))))) if type(min_probability) in (bool, int, float) else 0.01
            generalization = bool(generalization) if type(generalization) in (bool, int, float) else True
            stream = bool(stream) if type(stream) in (bool, int, float) else False
            interval = max((0.0, float(interval))) if type(interval) in (bool, int, float) else 0.0
            prompt, original_show_errors = rf'{prompt}', self.__show_errors
            self.__show_errors, probability, original_input, answer, adjusted, hurnet, fit_probability = False, 0.0, '', '', False, False, 0.0
            if self.__fine_tuning:
                standard_adjustment_forecast = self.__standardAdjustmentForecast(input_prompt=prompt, vector_matrix=self.__iterNumericVectors(model_path=self.__fit_model_path))
                probability, original_input, answer = standard_adjustment_forecast['probability'], standard_adjustment_forecast['prompt'], standard_adjustment_forecast['answer']
            hurnet_text, hurnet_probability = answer, 0.0
            if self.__hurnet_architecture and self.__hur_net is None and type(self.__state_hurnet) == dict and self.__state_hurnet != {}:
                self.__hur_net = self.__HurNet(device=self.__local_device, dtype=self.__float32)
                self.__hur_net.setParameters(state=self.__state_hurnet)
            if probability >= min_fit_probability: output_text, adjusted = answer, True
            elif self.__hurnet_architecture and self.__hur_net is not None:
                input_vector = self.__SCNet.textForEmbedding(text=self.__SCNet.normalization(input_text=prompt), length=self.__block_size, encoder=self.__tokenizer)
                output_vector = self.__hur_net.predict(input_layer=[input_vector], decimal_places=0)[0]
                output_vector = [abs(embedding) for embedding in output_vector]
                output_text = self.__SCNet.embeddingForText(embedding=output_vector, encoder=self.__tokenizer, strip=True)
                probability = self.__textualComparison(text1=prompt, text2=output_text, consider_length=False)
                hurnet_text, hurnet_probability = output_text, probability
                if probability >= min_fit_probability: adjusted, hurnet = True, len(output_text.strip()) > 0
                else: output_text = ''
            block_size = None if not self.__parallel_prediction else self.__block_size
            fit_probability, question_removal, self.__show_errors = probability, False, original_show_errors
            if len(output_text) < 1:
                if not self.__trained: return ''
                self.__SCNet.minimum_probability_for_candidates = self.minimum_probability_for_candidates
                if not self.__loaded and len(self.__inputs) > 0 and len(self.__outputs) > 0:
                    vector_matrix, new_embedding = [], self.__SCNet.textForEmbedding(text='\n', length=None, encoder=self.__tokenizer)
                    for _input, _output in zip(self.__inputs, self.__outputs): vector_matrix.append(_input+new_embedding+_output)
                    input_vector = self.__SCNet.textForEmbedding(text=prompt, length=None, encoder=self.__tokenizer)
                    _, output_vector = self.__vectorComparison(input_vector=input_vector, vector_matrix=vector_matrix)
                else:
                    input_vector = self.__SCNet.textForEmbedding(text=prompt, length=block_size, encoder=self.__tokenizer)
                    _, output_vector = self.__vectorComparison(input_vector=input_vector, vector_matrix=self.__iterNumericVectors())
                output_text = self.__SCNet.embeddingForText(embedding=output_vector, encoder=self.__tokenizer, strip=True)
                probability = self.__textualComparison(text1=prompt, text2=output_text, consider_length=True)
                if probability >= min_probability:
                    list_of_answers = []
                    self.__candidates = self.__SCNet._SCNet__candidates
                    if len(self.__candidates) > 0:
                        list_of_answers = [output_text]
                        for candidate in self.__candidates:
                            probabilities = self.__SCNet.similarityBetweenVectors(first_vector=input_vector, second_vector=candidate)
                            if probabilities >= self.minimum_probability_for_candidates: list_of_answers.append(self.__SCNet.embeddingForText(embedding=candidate, encoder=self.__tokenizer, strip=True))
                        from random import shuffle
                        shuffle(list_of_answers)
                        output_text = '\n\n'.join(list_of_answers)
                    output_text, self.__candidates = output_text.strip(), []
                else: output_text = ''
                if self.__SCNet.normalization(input_text=prompt) in self.__SCNet.normalization(input_text=output_text): generalization = False
                if len(output_text) > 0 and '?' in output_text and prompt.endswith('?'):
                    question_and_answer = output_text.split('?')
                    question, answer = question_and_answer[0].strip(), question_and_answer[-1].strip()
                    if len(question) > 0 and len(answer) > 0:
                        output_text, question_removal = output_text.replace(question, '').strip(), True
                        if output_text.startswith('?'): output_text = output_text[1:].strip()
                output_text_x = self.__extractResponse(prompt=prompt, text=output_text) if not question_removal else output_text
                if output_text_x != output_text: original_input = output_text.replace(output_text_x, '').strip()
            if adjusted and fit_probability > 0.7: generalization = False
            if not hurnet and (generalization and probability < 1.0): output_text = self.__SCNet.outputAdaptedToInput(prompt=prompt, original_input=original_input, original_output=output_text)
            if not question_removal and not adjusted and len(output_text) > 0: output_text = self.__extractResponse(prompt=prompt, text=output_text)
            if max_tokens is not None: output_text = self.__SCNet.truncateTokens(text=output_text, max_tokens=max_tokens, encoder=self.__tokenizer)
            output_text, self.fit_probability, self.probability = output_text.strip(), fit_probability, probability
            if hurnet_probability > probability: output_text, probability, self.probability = hurnet_text, hurnet_probability, hurnet_probability
            answer_dictionary = {'answer': output_text, 'fit_probability': fit_probability, 'probability': probability}
            if stream: return self.__getStream(final_answer=output_text, fit_probability=fit_probability, probability=probability, interval=interval)
            else: return answer_dictionary
        except Exception as error:
            if self.__show_errors: print('ERROR in __predict: '+str(error))
            if stream == True: return self.__getStream(final_answer='', probability=0.0, interval=0.0)
            else: return {'answer': '', 'fit_probability': 0.0, 'probability': 0.0}
    def __overwriteWithHexadecimal(self, file_path='', progress=False):
        try:
            file_path = str(file_path).strip()
            progress = bool(progress) if type(progress) in (bool, int, float) else False
            if not self.__exists(file_path):
                print(f'Overwrite: The path specified in {file_path} does not exist!!')
                return False
            temporary_file = self.__NamedTemporaryFile(delete=False, mode='w', encoding='utf-8', suffix='.temp')
            temporary_path = temporary_file.name
            chunk_size, current_line = self.__getBufferSize(), []
            total_limit = self.__getsize(file_path)
            self.__setTqdm(disable=not progress)
            progress_bar = self.__updateTqdm(total=total_limit, description='Encoding file')
            with open(file_path, 'rb') as input_file, open(temporary_path, 'w') as output_file:
                while True:
                    chunk = input_file.read(chunk_size)
                    if not chunk: break
                    for byte in chunk:
                        str_byte = f'{byte:04x}'
                        current_line.append(str_byte)
                        if len(current_line) == 8:
                            output_file.write(' '.join(current_line) + '\n')
                            current_line = []
                            progress_bar.update(len(str(str_byte).encode('utf-8'))*2)
                if current_line: output_file.write(' '.join(current_line))
            if progress_bar.n < total_limit: progress_bar.n = total_limit
            progress_bar.refresh()
            progress_bar.close()
            self.__replace(temporary_path, file_path)
            if self.__exists(temporary_path): self.__remove(temporary_path)
            return True
        except Exception as error:
            if self.__show_errors:
                print('ERROR in __overwriteWithHexadecimal: '+str(error))
                print('file_path: '+str(file_path))
            return False
    def __restoreOriginalFromHexadecimal(self, file_path='', progress=False):
        try:
            file_path = str(file_path).strip()
            progress = bool(progress) if type(progress) in (bool, int, float) else False
            if not self.__exists(file_path):
                print(f'Restore: The path specified in {file_path} does not exist!!')
                return False
            temporary_file = self.__NamedTemporaryFile(delete=False, mode='w', encoding='utf-8', suffix='.temp')
            temporary_path = temporary_file.name
            total_limit = self.__getsize(file_path)
            self.__setTqdm(disable=not progress)
            progress_bar = self.__updateTqdm(total=total_limit, description='Decoding file')
            with open(file_path, 'r', errors='ignore') as input_file, open(temporary_path, 'wb') as output_file:
                for line in input_file:
                    tokens = line.strip().split()
                    byte_values = bytes(int(token, 16) & 0xFF for token in tokens)
                    output_file.write(byte_values)
                    current_bytes = len(str(line).encode('utf-8'))
                    progress_bar.update(current_bytes)
            if progress_bar.n < total_limit: progress_bar.n = total_limit
            progress_bar.refresh()
            progress_bar.close()
            self.__replace(temporary_path, file_path)
            if self.__exists(temporary_path): self.__remove(temporary_path)
            return True
        except Exception as error:
            if self.__show_errors:
                print('ERROR in __restoreOriginalFromHexadecimal: '+str(error))
                print('file_path: '+str(file_path))
            return False
    def __fileToEncryptedHEX(self, file_path='', progress=False):
        try:
            file_path = str(file_path).strip()
            progress = bool(progress) if type(progress) in (bool, int, float) else False
            if not self.__exists(file_path):
                print(f'Overwrite: The path specified in {file_path} does not exist!!')
                return False
            temporary_file = self.__NamedTemporaryFile(delete=False, mode='w', encoding='utf-8', suffix='.temp')
            temporary_path = temporary_file.name
            chunk_size, current_line = self.__getBufferSize(), []
            total_limit = self.__getsize(file_path)
            self.__setTqdm(disable=not progress)
            progress_bar = self.__updateTqdm(total=total_limit, description='Encoding file')
            with open(file_path, 'rb') as input_file, open(temporary_path, 'w') as output_file:
                while True:
                    chunk = input_file.read(chunk_size)
                    if not chunk: break
                    for byte in chunk:
                        str_byte = f'{byte:04x}'[::-1]
                        current_line.append(str_byte)
                        if len(current_line) == 8:
                            output_file.write(' '.join(current_line[::-1]) + '\n')
                            current_line = []
                            progress_bar.update(len(str(str_byte).encode('utf-8'))*2)
                if current_line: output_file.write(' '.join(current_line[::-1]))
            if progress_bar.n < total_limit: progress_bar.n = total_limit
            progress_bar.refresh()
            progress_bar.close()
            self.__replace(temporary_path, file_path)
            if self.__exists(temporary_path): self.__remove(temporary_path)
            return True
        except Exception as error:
            if self.__show_errors:
                print('ERROR in __fileToEncryptedHEX: '+str(error))
                print('file_path: '+str(file_path))
            return False
    def __encryptedHEXToFile(self, file_path='', progress=False):
        try:
            file_path = str(file_path).strip()
            progress = bool(progress) if type(progress) in (bool, int, float) else False
            if not self.__exists(file_path):
                print(f'Restore: The path specified in {file_path} does not exist!!')
                return False
            temporary_file = self.__NamedTemporaryFile(delete=False, mode='w', encoding='utf-8', suffix='.temp')
            temporary_path = temporary_file.name
            total_limit = self.__getsize(file_path)
            self.__setTqdm(disable=not progress)
            progress_bar = self.__updateTqdm(total=total_limit, description='Decoding file')
            with open(file_path, 'r', errors='ignore') as input_file, open(temporary_path, 'wb') as output_file:
                for line in input_file:
                    tokens = line.strip().split()[::-1]
                    byte_values = bytes(int(token[::-1], 16) & 0xFF for token in tokens)
                    output_file.write(byte_values)
                    current_bytes = len(str(line).encode('utf-8'))
                    progress_bar.update(current_bytes)
            if progress_bar.n < total_limit: progress_bar.n = total_limit
            progress_bar.refresh()
            progress_bar.close()
            self.__replace(temporary_path, file_path)
            if self.__exists(temporary_path): self.__remove(temporary_path)
            return True
        except Exception as error:
            if self.__show_errors:
                print('ERROR in __encryptedHEXToFile: '+str(error))
                print('file_path: '+str(file_path))
            return False
    def __getTrainingText(self, max_tokens=None):
        try:
            result_text = ''
            max_tokens, tokens_count = int(max_tokens) if type(max_tokens) in (bool, int, float) else None, 0
            if self.__exists(self.__temporary_path):
                for embedding in self.__iterNumericVectors():
                    text = self.__SCNet.embeddingForText(embedding=embedding, encoder=self.__tokenizer, strip=True)
                    if '<|' in text or '|>' in text:
                        def _remove_between_markers(text=''): return self.__sub(r'<\|.*?\|>', '', text, flags=self.__DOTALL)
                        text = _remove_between_markers(text=text)
                    result_text += text.strip()+'\n\n'
                    tokens_count += self.countTokens(text=result_text, encoder=self.__tokenizer)
                    if max_tokens is not None and tokens_count >= max_tokens: return self.__SCNet.textSummary(text=result_text, max_tokens=max_tokens, encoder=self.__tokenizer)
            if self.__exists(self.__fit_model_path):
                for embedding in self.__iterNumericVectors(model_path=self.__fit_model_path):
                    text = self.__SCNet.embeddingForText(embedding=embedding, encoder=self.__tokenizer, strip=True)
                    pairs = text.strip().split('<|end|>')
                    question, answer = pairs[-2].strip(), pairs[-1].strip()
                    result_text += f'{question}\n{answer}\n\n'
                    tokens_count += self.__SCNet.countTokens(text=result_text, encoder=self.__tokenizer)
                    if max_tokens is not None and tokens_count >= max_tokens: return self.__SCNet.textSummary(text=result_text, max_tokens=max_tokens, encoder=self.__tokenizer)
            return result_text.strip()
        except Exception as error:
            if self.__show_errors: print('ERROR in __getTrainingText: '+str(error))
            return ''
    def __addHiddenLayer(self, num_neurons=0, activation_function='linear'):
        try:
            num_neurons = int(num_neurons) if type(num_neurons) in (bool, int, float) else 0
            activation_function = str(activation_function).lower().strip()
            self.__hidden_layers.append((num_neurons, activation_function))
            return len(self.__hidden_layers) > 0
        except Exception as error:
            if self.__show_errors: print('ERROR in __addHiddenLayer: '+str(error))
            return False
    def __train(self, dataset_path='', string='', precision=1.0, tokenizer='gpt-4', context_window=float('inf'), end_tag='\n\n', validate=0.0, progress=True):
        try:
            training_metrics, infinite = {'val_loss': 0.0, 'loss': 1.0, 'generalization_rate': 0.0, 'precision': 0.0}, float('inf')
            dataset_path = str(dataset_path).strip()
            string = str(string).strip()
            precision = min((1.0, max((0.0, float(precision))))) if type(precision) in (bool, int, float) else 1.0
            tokenizer = str(tokenizer).lower().strip()
            context_window = max((1, float(context_window))) if type(context_window) in (bool, int, float) else infinite
            end_tag = str(end_tag)
            validate = min((1.0, max((0.0, float(validate))))) if type(validate) in (bool, int, float) else 0.0
            progress = bool(progress) if type(progress) in (bool, int, float) else True
            string = rf'{string}'
            self.__setTqdm(disable=not progress)
            progress_bar = self.__updateTqdm(total=7, description='Training model')
            if len(end_tag) < 1: end_tag = '\n\n'
            progress_bar.update(1)
            self.__string = str(self.__string).strip()
            progress_bar.update(1)
            if len(self.__string) > 0: self.__string = self.__string.replace('<|context|>', '\n').replace('<|end|>', end_tag).strip()
            progress_bar.update(1)
            string = str(self.__string+'\n\n'+string).strip()
            progress_bar.update(1)
            self.__temporary_path = self.__processDataset(dataset_path=dataset_path, string=string, precision=precision, tokenizer=tokenizer, context_window=context_window, end_tag=end_tag, progress=progress)
            if validate > 0 and self.__exists(self.__temporary_path):
                probabilities_x, probabilities_y = [], []
                total_bytes, bytes_count = self.__getsize(self.__temporary_path), 0
                if total_bytes > 0:
                    bytes_limit = int(total_bytes * validate)
                    progress_bar_val = self.__updateTqdm(total=bytes_limit, description='Validating model')
                    for token_generator in self.__iterNumericVectors(model_path=self.__temporary_path):
                        prompt = answer = input_output = self.__SCNet.embeddingForText(embedding=token_generator, encoder=tokenizer, strip=True)
                        for separator in self.__separators:
                            if separator in input_output:
                                chunks = input_output.strip().split(separator)
                                prompt = chunks[0].strip()+separator
                                chunks = chunks[1:]
                                answer = separator.join(chunks).strip()
                                break
                        input_vector = self.__SCNet.textForEmbedding(text=prompt, length=self.__block_size, encoder=tokenizer)
                        output_vector = self.__SCNet.textForEmbedding(text=answer, length=self.__block_size, encoder=tokenizer)
                        _, approximate_output = self.__vectorComparison(input_vector=input_vector, vector_matrix=self.__iterNumericVectors(model_path=self.__temporary_path))
                        approximate_text = self.__SCNet.embeddingForText(embedding=approximate_output, encoder=tokenizer, strip=True)
                        approximate_text = self.__extractResponse(prompt=prompt, text=approximate_text)
                        approximate_output = self.__SCNet.textForEmbedding(text=approximate_text, length=self.__block_size, encoder=tokenizer)
                        probability_x = self.__SCNet.similarityBetweenVectors(first_vector=output_vector, second_vector=approximate_output)
                        probability_y = self.__SCNet.euclideanSimilarity(first_vector=output_vector, second_vector=approximate_output)
                        probabilities_x.append(probability_x), probabilities_y.append(1.0-((probability_x+probability_y)/2))
                        token_generator_str = ' '.join(map(str, token_generator))+'\n'
                        current_bytes = len(token_generator_str.encode('utf-8'))
                        progress_bar_val.update(current_bytes)
                        bytes_count += current_bytes
                        if bytes_count >= bytes_limit:
                            progress_bar_val.n = bytes_limit
                            progress_bar_val.refresh()
                            break
                    progress_bar_val.close()
                    loss_probability = float(self.__mean(probabilities_x))
                    val_loss = float(self.__mean(probabilities_y))
                    loss = 1.0-loss_probability
                    generalization_rate = 1.0-val_loss
                    training_metrics['val_loss'] = val_loss
                    training_metrics['loss'] = loss
                    training_metrics['generalization_rate'] = generalization_rate
                    training_metrics['precision'] = loss_probability
            else: training_metrics = {'val_loss': 0.0, 'loss': 1.0-precision, 'generalization_rate': precision/2, 'precision': precision}
            progress_bar.update(1)
            self.__tokenizer = tokenizer
            progress_bar.update(1)
            self.__trained = True
            progress_bar.update(1)
            progress_bar.close()
            self.__setTqdm(disable=True)
            self.__training_finished = True
            return training_metrics
        except Exception as error:
            if self.__show_errors: print('ERROR in __train: '+str(error))
            try:
                self.__setTqdm(disable=True)
                return training_metrics
            except: return {'val_loss': 1.0, 'loss': 1.0, 'generalization_rate': 0.0, 'precision': 0.0}
    def __saveModel(self, model_path='', progress=True):
        try:
            model_path = file_name = str(model_path).strip()
            progress = bool(progress) if type(progress) in (bool, int, float) else True
            original_model_path, self.__progress = model_path, progress
            self.__setTqdm(disable=not progress)
            progress_bar = self.__updateTqdm(total=5, description='Saving model')
            progress_bar.update(1)
            model_path = self.__changeExtension(model_path=model_path, old_extension='scn02', new_extension='scn01')
            if len(model_path) > 0:
                directory, file_name = self.__split(model_path)
                if not file_name: file_name = 'model.scn01'
                elif not file_name.endswith('.scn01'):
                    point_count, index = file_name.count('.'), file_name.find('.')
                    if point_count > 1: file_name = file_name[:file_name.rfind('.')]
                    elif point_count > 0 and index > 0: file_name = file_name[:index]
                    file_name += '.scn01'
            else: directory, file_name = str(model_path), 'model.scn01'
            progress_bar.update(1)
            if directory and not self.__exists(directory): self.__makedirs(directory)
            save_path = self.__join(directory, file_name)
            save_dict = {
                'tokenizer': str(self.__tokenizer).lower().strip(),
                'block_size': max((1, int(self.__block_size))) if type(self.__block_size) in (bool, int, float) else 50,
                'parameters_number': max((0, int(self.parameters_number))) if type(self.parameters_number) in (bool, int, float) else 0,
                'parallel_prediction': int(self.__parallel_prediction) if type(self.__parallel_prediction) in (bool, int, float) else 0,
                'minimum_probability_for_candidates': float(self.minimum_probability_for_candidates) if type(self.minimum_probability_for_candidates) in (bool, int, float) else 0.9,
                'state_hurnet': self.__state_hurnet,
                'model_state_dict': []

            }
            progress_bar.update(1)
            if len(self.__temporary_path.strip()) > 0 and not self.__add_hur:
                destination_path = self.__changeExtension(model_path=save_path, old_extension='scn01', new_extension='scn02')
                formatted_params = self.__format_numbers(data_number=self.parameters_number, is_tokens=False)
                self.__moveAndRenameStream(source_path=self.__temporary_path, destination_path=destination_path, formatted_params=formatted_params, progress=progress)
                self.__overwriteWithHexadecimal(file_path=destination_path, progress=self.__progress)
            self.__save(save_dict, save_path)
            progress_bar.update(1)
            self.__trained = True
            def _copy_model_files():
                try:
                    new_directory = directory
                    old_directory = self.__dirname(self.__fit_model_path)
                    old_file_name = self.__splitext(self.__basename(self.__fit_model_path))[0]
                    names = self.__splitext(self.__basename(file_name))[0]
                    self.__makedirs(new_directory, exist_ok=True)
                    for filename in self.__listdir(old_directory):
                        if filename.startswith(old_file_name):
                            old_path = self.__join(old_directory, filename)
                            if self.__isfile(old_path):
                                _, extension = self.__splitext(filename)
                                new_filename = names + extension
                                new_path = self.__join(new_directory, new_filename)
                                if old_path != new_path:
                                    self.__copy2(old_path, new_path)
                                    self.__overwriteWithHexadecimal(file_path=new_path, progress=self.__progress)
                    return True
                except: return False
            copy_model_files = _copy_model_files() if self.__hurnet_architecture else True
            progress_bar.update(1)
            progress_bar.close()
            if self.__loaded and self.__training_finished:
                continuous_pre_training = False
                progress_bar = self.__updateTqdm(total=3, description='Continuous Pre-training')
                progress_bar.update(1)
                transmitter_path, receiver_path = self.__path_of_the_loaded_model, original_model_path
                transfer_learning_scn02 = self.__transferLearning(transmitter_path=transmitter_path, receiver_path=receiver_path, extension='scn02', continuous_pre_training=True, progress=progress)
                progress_bar.update(1)
                if transfer_learning_scn02:
                    transfer_learning_scn01 = self.__transferLearning(transmitter_path=transmitter_path, receiver_path=receiver_path, extension='scn01', continuous_pre_training=True, progress=progress)
                    transfer_learning_scn03 = self.__transferLearning(transmitter_path=transmitter_path, receiver_path=receiver_path, extension='scn03', continuous_pre_training=True, progress=progress)
                    continuous_pre_training = transfer_learning_scn01 or transfer_learning_scn02 or transfer_learning_scn03
                progress_bar.update(1)
                return continuous_pre_training
            return copy_model_files
        except Exception as error:
            if self.__show_errors: print('ERROR in __saveModel: '+str(error))
            self.__setTqdm(disable=True)
            return False
    def __loadModel(self, model_path='', progress=True):
        try:
            model_path = str(model_path).strip()
            progress = bool(progress) if type(progress) in (bool, int, float) else True
            self.__path_of_the_loaded_model, self.__progress = model_path, progress
            self.__setTqdm(disable=not progress)
            progress_bar = self.__updateTqdm(total=7, description='Loading model')
            progress_bar.update(1)
            model_path = self.__changeExtension(model_path=model_path, old_extension='scn02', new_extension='scn01')
            if len(model_path) > 0:
                directory, file_name = self.__split(model_path)
                if not file_name: file_name = 'model.scn01'
                elif not file_name.endswith('.scn01'):
                    point_count, index = file_name.count('.'), file_name.find('.')
                    if point_count > 1: file_name = file_name[:file_name.rfind('.')]
                    elif point_count > 0 and index > 0: file_name = file_name[:index]
                    file_name += '.scn01'
            else:
                def _find_first_scn01_file(directory='.'):
                    directory_path = self.__Path(directory)
                    if not directory_path.is_dir(): return 'model.scn01'
                    scn01_file_list = sorted(file_path for file_path in directory_path.iterdir() if file_path.is_file() and file_path.suffix.lower() == '.scn01')
                    return scn01_file_list[0] if scn01_file_list else 'model.scn01'
                directory, file_name = '', _find_first_scn01_file()
            model_path = self.__join(directory, file_name)
            progress_bar.update(1)
            if not self.__exists(model_path):
                progress_bar.update(5)
                return False
            try: checkpoint = self.__load(model_path, map_location='cuda' if self.__cuda.is_available() else 'cpu')
            except: checkpoint = self.__load(model_path, map_location='cpu')
            try: self.__tokenizer = str(checkpoint['tokenizer']).lower().strip()
            except: self.__tokenizer = 'gpt-4'
            try: self.__block_size = max((1, int(checkpoint['block_size'])))
            except: self.__block_size = 50
            try: self.parameters_number = max((0, int(checkpoint['parameters_number']))) if type(checkpoint['parameters_number']) in (bool, int, float) else 0
            except: self.parameters_number = 0
            try: parallel_prediction = bool(checkpoint['parallel_prediction']) if type(checkpoint['parallel_prediction']) in (bool, int, float) else False
            except: parallel_prediction = False
            if parallel_prediction: self.__parallel_prediction = parallel_prediction
            try: self.minimum_probability_for_candidates = float(checkpoint['minimum_probability_for_candidates']) if type(checkpoint['minimum_probability_for_candidates']) in (bool, int, float) else 0.9
            except: self.minimum_probability_for_candidates = 0.9
            try: self.__state_hurnet = checkpoint['state_hurnet']
            except: self.__state_hurnet = {}
            self.__SCNet = self.__load_SCNet(
                device=self.__local_device,
                user_id=self.user_id,
                parallel_prediction=self.__parallel_prediction,
                minimum_probability_for_candidates=self.minimum_probability_for_candidates,
                show_errors=False
            )
            progress_bar.update(1)
            model_path = self.__changeExtension(model_path=model_path, old_extension='scn01', new_extension='scn02')
            self.__restoreOriginalFromHexadecimal(file_path=model_path, progress=self.__progress)
            progress_bar.update(1)
            self.__fit_model_path = self.__changeExtension(model_path=model_path, old_extension='scn02', new_extension='scn03')
            if self.__exists(self.__fit_model_path): self.__restoreOriginalFromHexadecimal(file_path=self.__fit_model_path, progress=self.__progress)
            progress_bar.update(1)
            if self.__exists(self.__fit_model_path): self.__fine_tuning = True
            if type(self.__state_hurnet) == dict and self.__state_hurnet != {}:
                self.__hur_net = self.__HurNet(device=self.__local_device, dtype=self.__float32)
                self.__hur_net.setParameters(state=self.__state_hurnet)
                self.__hurnet_architecture = True
            progress_bar.update(1)
            self.__temporary_path, self.__model_path, self.__trained, self.__loaded = model_path, model_path, True, True
            progress_bar.update(1)
            progress_bar.close()
            return True
        except Exception as error:
            if self.__show_errors: print('ERROR in __loadModel: '+str(error))
            self.__setTqdm(disable=True)
            return False
    def __transferLearning(self, transmitter_path='', receiver_path='', extension='scn02', continuous_pre_training=False, progress=True):
        try:
            transmitter_path, receiver_path, extension = str(transmitter_path).strip(), str(receiver_path).strip(), str(extension).lower().strip()
            continuous_pre_training = bool(continuous_pre_training) if type(continuous_pre_training) in (bool, int, float) else False
            progress = bool(progress) if type(progress) in (bool, int, float) else True
            self.__progress = progress
            def _merge_scn02_files(transmitter_path='', receiver_path=''):
                def __resolve_path(input_path=''):
                    if self.__isdir(input_path):
                        matching_files = self.__glob(self.__join(input_path, f'*.{extension}'))
                        return matching_files[0] if matching_files else None
                    elif not input_path.lower().endswith(f'.{extension}'): input_path += f'.{extension}'
                    if self.__isfile(input_path):
                        if input_path.lower().endswith(f'.{extension}'): return input_path
                        parent_directory = self.__dirname(input_path)
                        matching_files = self.__glob(self.__join(parent_directory, f'*.{extension}'))
                        return matching_files[0] if matching_files else None
                    return None
                resolved_transmitter_path = __resolve_path(transmitter_path)
                resolved_receiver_path = __resolve_path(receiver_path)
                def _copy_file(transmitter_path='', receiver_path=''):
                    if not self.__exists(transmitter_path): return False
                    if not self.__exists(receiver_path):
                        self.__copyfile(transmitter_path, receiver_path)
                        return True
                    if self.__getsize(transmitter_path) > self.__getsize(receiver_path):
                        self.__copyfile(transmitter_path, receiver_path)
                        return True
                    return False
                if resolved_transmitter_path is None: return False
                elif resolved_receiver_path is None:
                    parent_directory = self.__dirname(receiver_path) if self.__isfile(receiver_path) else receiver_path
                    resolved_receiver_path = [path for pattern in ('*.scn01', '*.sccon', '*.scconf') for path in self.__glob(self.__join(parent_directory, pattern))][0]
                    if resolved_receiver_path.endswith('.scn01'): old_extension = 'scn01'
                    elif resolved_receiver_path.endswith('.sccon'): old_extension = 'sccon'
                    elif resolved_receiver_path.endswith('.scconf'): old_extension = 'scconf'
                    resolved_receiver_path = self.__changeExtension(model_path=resolved_receiver_path, old_extension=old_extension, new_extension=extension)
                    return _copy_file(transmitter_path=resolved_transmitter_path, receiver_path=resolved_receiver_path)
                elif extension == 'scn01': return _copy_file(transmitter_path=resolved_transmitter_path, receiver_path=resolved_receiver_path)
                restore_original_from_hexadecimal = self.__restoreOriginalFromHexadecimal(file_path=resolved_transmitter_path, progress=self.__progress) if not continuous_pre_training else True
                if restore_original_from_hexadecimal:
                    if self.__restoreOriginalFromHexadecimal(file_path=resolved_receiver_path, progress=self.__progress):
                        if not resolved_transmitter_path or not resolved_receiver_path: return False
                        buffer_size = self.__getBufferSize()
                        total_bytes, bytes_count = self.__getsize(resolved_transmitter_path), 0
                        progress_bar = self.__updateTqdm(total=total_bytes, description=f'Transferring learn - {extension}')
                        with open(resolved_receiver_path, 'ab') as destination_file, open(resolved_transmitter_path, 'rb') as source_file:
                            for data_chunk in iter(lambda: source_file.read(buffer_size), b''):
                                destination_file.write(data_chunk)
                                current_bytes = buffer_size
                                progress_bar.update(current_bytes)
                                bytes_count += current_bytes
                            progress_bar.n = total_bytes
                            progress_bar.refresh()
                            progress_bar.close()
                        overwrite_with_hexadecimal = self.__overwriteWithHexadecimal(file_path=resolved_transmitter_path, progress=self.__progress) if not continuous_pre_training else True
                        if overwrite_with_hexadecimal: return self.__overwriteWithHexadecimal(file_path=resolved_receiver_path, progress=self.__progress)
                return False
            return _merge_scn02_files(transmitter_path=transmitter_path, receiver_path=receiver_path)
        except Exception as error:
            if self.__show_errors: print('ERROR in __transferLearning: '+str(error))
            self.__setTqdm(disable=True)
            return False
    def __addFit(self, prompt='', answer='', id=0, relationship_id=0):
        try:
            prompt, answer = str(prompt).strip(), str(answer).strip()
            prompt, answer = rf'{prompt}', rf'{answer}'
            id = int(id) if type(id) in (bool, int, float) else 0
            relationship_id = int(relationship_id) if type(relationship_id) in (bool, int, float) else 0
            if self.__trained:
                if not self.__loaded: return self.__setInputsAndOutputs(prompt=prompt, answer=answer)
                if len(self.__hidden_layers) > 0 and (id <= 0 and relationship_id <= 0):
                    self.__add_hur = True
                    self.__setInputsAndOutputs(prompt=prompt, answer=answer)
                else:
                    if len(prompt) < 1 or len(answer) < 1: return False
                    input_output = str(id)+'<|end|>'+str(relationship_id)+'<|end|>'+prompt+'<|end|>'+answer
                    model_path = self.__fit_model_path
                    state, restore = 'a' if self.__exists(model_path) else 'w', True
                    if state == 'a' and self.__add_fit: restore = self.__restoreOriginalFromHexadecimal(file_path=model_path, progress=self.__progress)
                    if restore:
                        write = False
                        with open(model_path, state, encoding='utf-8') as open_file:
                            embeddings = self.__SCNet.textForEmbedding(text=input_output, length=None, encoder=self.__tokenizer)
                            open_file.write(' '.join(map(str, embeddings))+'\n')
                            write = True
                        if write: self.__overwriteWithHexadecimal(file_path=model_path, progress=self.__progress)
                        self.__add_fit = True
            else: self.__string += prompt+'<|context|>'+answer+'<|end|>'
            return True
        except Exception as error:
            if self.__show_errors: print('ERROR in __addFit: '+str(error))
            return False
    def __fit(self, activation_function='linear', bias=0.0, learning_rate=1.0, quantization=None, method='division', progress=True):
        try:
            method = str(method).lower().strip()
            if len(self.__hidden_layers) <= 0: return False
            progress = bool(progress) if type(progress) in (bool, int, float) else True
            self.__setTqdm(disable=not progress)
            progress_bar = self.__updateTqdm(total=4, description='Fitting model')
            progress_bar.update(1)
            if not self.__trained and len(self.__string) > 0 and '<|end|>' in self.__string:
                inputs_outputs = self.__string.split('<|end|>')
                progress_bar_text = self.__updateTqdm(total=len(inputs_outputs), description='Converting data')
                x, y, main_x_y = [], [], 0
                for input_output in inputs_outputs:
                    if len(input_output.strip()) > 0 and '<|context|>' in input_output:
                        prompt, answer = input_output.split('<|context|>')
                        x.append(prompt), y.append(answer)
                        max_x_y = max((len(prompt.split()), len(answer.split())))
                        if max_x_y > main_x_y: main_x_y = max_x_y
                    progress_bar_text.update(1)
                    progress_bar_text.close()
                self.__block_size = main_x_y*2
                for a, b in zip(x, y): self.__setInputsAndOutputs(prompt=a, answer=b)
            progress_bar.update(1)
            if len(self.__inputs) < 1 or len(self.__outputs) < 1:
                progress_bar.update(2)
                return False
            hur_net = self.__HurNet(device=self.__local_device, dtype=self.__float32)
            for num_neurons, activation_function in self.__hidden_layers: hur_net.addHiddenLayer(num_neurons=num_neurons, activation_function=activation_function)
            train_result = hur_net.train(input_layer=self.__inputs, output_layer=self.__outputs, activation_function=activation_function, bias=bias, learning_rate=learning_rate, quantization=quantization, method=method)
            progress_bar.update(1)
            self.__state_hurnet = hur_net.getParameters()
            self.__hurnet_architecture = True
            progress_bar.update(1)
            progress_bar.close()
            return train_result
        except Exception as error:
            if self.__show_errors: print('ERROR in __fit: '+str(error))
            self.__setTqdm(disable=True)
            return False
    def __print_predict(self, prompt='', max_tokens=None, min_fit_probability=0.7, min_probability=0.01, generalization=True, stream=False, interval=0.0):
        try:
            if stream == True:
                tokens_generator = self.__predict(prompt=prompt, max_tokens=max_tokens, min_fit_probability=min_fit_probability, min_probability=min_probability, generalization=generalization, stream=stream, interval=interval)
                for answer_dictionary in tokens_generator:
                    answer = answer_dictionary['answer']
                    print(answer, end='', flush=True)
                print()
            else: print(self.__predict(prompt=prompt, max_tokens=max_tokens, min_fit_probability=min_fit_probability, min_probability=min_probability, generalization=generalization, stream=stream, interval=interval)['answer'])
        except Exception as error:
            if self.__show_errors: print('ERROR in __print_predict: '+str(error))
    def __downloadFile(self, url_path='', file_path='', progress=True):
        try:
            url_path, file_path = str(url_path).strip(), str(file_path).strip()
            progress = bool(progress) if type(progress) in (bool, int, float) else True
            if len(url_path) < 1 or not url_path.lower().startswith(('http://', 'https://', 'www.')): return False
            original_file_name = self.__basename(self.__urlparse(url_path).path)
            if not file_path: target_file_path = original_file_name
            else:
                candidate_name = self.__basename(file_path)
                if not candidate_name or file_path.endswith('/') or file_path.endswith('\\'): target_file_path = self.__join(file_path, original_file_name)
                else: target_file_path = file_path
            directory = self.__dirname(target_file_path)
            if directory: self.__makedirs(directory, exist_ok=True)
            if self.__isdir(target_file_path) and not target_file_path.endswith('/') and not target_file_path.endswith('\\'): target_file_path = file_path+'/'+original_file_name
            response = self.__urlopen(url_path)
            total_size, block_size = int(response.getheader('Content-Length', 0)), 1024
            self.__setTqdm(disable=not progress)
            with open(target_file_path, 'wb') as file_handle, self.__tqdm(total=total_size, unit='B', unit_scale=True, desc=target_file_path, ncols=self.__get_terminal_size().columns) as progress_bar:
                while True:
                    buffer = response.read(block_size)
                    if not buffer: break
                    file_handle.write(buffer)
                    progress_bar.update(len(buffer))
                if progress_bar.n < total_size: progress_bar.n = total_size
                progress_bar.refresh()
                progress_bar.close()
            return self.__exists(target_file_path)
        except Exception as error:
            print('ERROR in __downloadFile: '+str(error))
            self.__setTqdm(disable=False)
            return False
    def downloadFile(self, url_path='', file_path='', progress=True):
        try: return self.__downloadFile(url_path=url_path, file_path=file_path, progress=progress)
        except Exception as error:
            if self.__show_errors: print('ERROR in downloadFile: '+str(error))
            return False
    def overwriteWithHexadecimal(self, file_path='', progress=False):
        try: return self.__overwriteWithHexadecimal(file_path=file_path, progress=progress)
        except Exception as error:
            if self.__show_errors: print('ERROR in overwriteWithHexadecimal: '+str(error))
            return False
    def restoreOriginalFromHexadecimal(self, file_path='', progress=False):
        try: return self.__restoreOriginalFromHexadecimal(file_path=file_path, progress=progress)
        except Exception as error:
            if self.__show_errors: print('ERROR in restoreOriginalFromHexadecimal: '+str(error))
            return False
    def fileToEncryptedHEX(self, file_path='', progress=False):
        try: return self.__fileToEncryptedHEX(file_path=file_path, progress=progress)
        except Exception as error:
            if self.__show_errors: print('ERROR in fileToEncryptedHEX: '+str(error))
            return False
    def encryptedHEXToFile(self, file_path='', progress=False):
        try: return self.__encryptedHEXToFile(file_path=file_path, progress=progress)
        except Exception as error:
            if self.__show_errors: print('ERROR in encryptedHEXToFile: '+str(error))
            return False
    def loadJSON(self, file_path='', string_content='', key_name=''):
        try:
            key_name = str(key_name).strip()
            self.__SCNet._SCNet__show_errors = self.__show_errors
            json_loaded = self.__SCNet.loadJSON(file_path=file_path, string_content=string_content)
            return json_loaded[key_name] if len(key_name) > 0 else json_loaded
        except Exception as error:
            print('ERROR in loadJSON: '+str(error))
            return {}
    def normalization(self, input_text=''):
        try:
            self.__SCNet._SCNet__show_errors = self.__show_errors
            return self.__SCNet.normalization(input_text=input_text)
        except Exception as error:
            if self.__show_errors: print('ERROR in normalization: '+str(error))
            return input_text
    def countTokens(self, text='', encoder='gpt-4'):
        try:
            self.__SCNet._SCNet__show_errors = self.__show_errors
            return self.__SCNet.countTokens(text=text, encoder=encoder)
        except Exception as error:
            if self.__show_errors: print('ERROR in countTokens: '+str(error))
            return len(str(text))
    def truncateTokens(self, text='', max_tokens=1000000, encoder='gpt-4'):
        try:
            self.__SCNet._SCNet__show_errors = self.__show_errors
            return self.__SCNet.truncateTokens(text=text, max_tokens=max_tokens, encoder=encoder)
        except Exception as error:
            if self.__show_errors: print('ERROR in truncateTokens: '+str(error))
            return text
    def textForEmbedding(self, text='', length=None, encoder='gpt-4'):
        try:
            self.__SCNet._SCNet__show_errors = self.__show_errors
            return self.__SCNet.textForEmbedding(text=text, length=length, encoder=encoder)
        except Exception as error:
            if self.__show_errors: print('ERROR in textForEmbedding: '+str(error))
            return []
    def embeddingForText(self, embedding=[], encoder='gpt-4', strip=True):
        try:
            self.__SCNet._SCNet__show_errors = self.__show_errors
            return self.__SCNet.embeddingForText(embedding=embedding, encoder=encoder, strip=strip)
        except Exception as error:
            if self.__show_errors: print('ERROR in embeddingForText: '+str(error))
            return ''
    def textSummary(self, text='', max_tokens=1000000, encoder='gpt-4'):
        try:
            self.__SCNet._SCNet__show_errors = self.__show_errors
            return self.__SCNet.textSummary(text=text, max_tokens=max_tokens, encoder=encoder)
        except Exception as error:
            if self.__show_errors: print('ERROR in textSummary: '+str(error))
            return text
    def getTrainingText(self, max_tokens=None):
        try: return self.__getTrainingText(max_tokens=max_tokens)
        except Exception as error:
            if self.__show_errors: print('ERROR in getTrainingText: '+str(error))
            return ''
    def alternativeVectorComparison(self, input_vector=[], vector_matrix=[]):
        try:
            self.__SCNet._SCNet__show_errors = self.__show_errors
            return self.__SCNet.alternativeVectorComparison(input_vector=input_vector, vector_matrix=vector_matrix)
        except Exception as error:
            if self.__show_errors: print('ERROR in alternativeVectorComparison: '+str(error))
            return (-1, [])
    def vectorComparison(self, input_vector=[], vector_matrix=[], stream=True):
        try:
            stream = bool(stream) if type(stream) in (bool, int, float) else True
            def _matrix_loop():
                for vector in vector_matrix: yield vector
            self.__SCNet._SCNet__show_errors = self.__show_errors
            return self.__vectorComparison(input_vector=input_vector, vector_matrix=_matrix_loop() if stream else vector_matrix)
        except Exception as error:
            if self.__show_errors: print('ERROR in vectorComparison: '+str(error))
            return (-1, [])
    def textualComparison(self, text1='', text2='', consider_length=True):
        try: return self.__textualComparison(text1=text1, text2=text2, consider_length=consider_length)
        except Exception as error:
            print('ERROR in textualComparison: ' + str(error))
            return 0.0
    def similarityBetweenVectors(self, first_vector=[], second_vector=[]):
        try:
            self.__SCNet._SCNet__show_errors = self.__show_errors
            return self.__SCNet.similarityBetweenVectors(first_vector=first_vector, second_vector=second_vector)
        except Exception as error:
            if self.__show_errors: print('ERROR in similarityBetweenVectors: '+str(error))
            return 0.0
    def euclideanSimilarity(self, first_vector=[], second_vector=[]):
        try:
            self.__SCNet._SCNet__show_errors = self.__show_errors
            return self.__SCNet.euclideanSimilarity(first_vector=first_vector, second_vector=second_vector)
        except Exception as error:
            if self.__show_errors: print('ERROR in euclideanSimilarity: '+str(error))
            return 0.0
    def outputAdaptedToInput(self, prompt='', original_input='', original_output=''):
        try:
            self.__SCNet._SCNet__show_errors = self.__show_errors
            return self.__SCNet.outputAdaptedToInput(prompt=prompt, original_input=original_input, original_output=original_output)
        except Exception as error:
            if self.__show_errors: print('ERROR in outputAdaptedToInput: '+str(error))
            return original_output
    def addHiddenLayer(self, num_neurons=0, activation_function='linear'):
        try: return self.__addHiddenLayer(num_neurons=num_neurons, activation_function=activation_function)
        except Exception as error:
            if self.__show_errors: print('ERROR in addHiddenLayer: '+str(error))
            return False
    def train(self, dataset_path='', string='', precision=1.0, tokenizer='gpt-4', context_window=float('inf'), end_tag='\n\n', validate=0.0, progress=True):
        try: return self.__train(dataset_path=dataset_path, string=string, precision=precision, tokenizer=tokenizer, context_window=context_window, end_tag=end_tag, validate=validate, progress=progress)
        except Exception as error:
            if self.__show_errors: print('ERROR in train: '+str(error))
            return {'val_loss': 1.0, 'loss': 1.0, 'generalization_rate': 0.0, 'precision': 0.0}
    def saveModel(self, model_path='', progress=True):
        try: return self.__saveModel(model_path=model_path, progress=progress)
        except Exception as error:
            if self.__show_errors: print('ERROR in saveModel: '+str(error))
            return False
    def loadModel(self, model_path='', progress=True):
        try: return self.__loadModel(model_path=model_path, progress=progress)
        except Exception as error:
            if self.__show_errors: print('ERROR in loadModel: '+str(error))
            return False
    def transferLearning(self, transmitter_path='', receiver_path='', progress=True):
        try:
            transfer_learning_scn01, transfer_learning_scn02, transfer_learning_scn03 = False, False, False
            transfer_learning_scnet, transfer_learning_scfit, transfer_learning_hurnet = False, False, False
            transfer_learning_scn02 = self.__transferLearning(transmitter_path=transmitter_path, receiver_path=receiver_path, extension='scn02', continuous_pre_training=False, progress=progress)
            if transfer_learning_scn02:
                transfer_learning_scn01 = self.__transferLearning(transmitter_path=transmitter_path, receiver_path=receiver_path, extension='scn01', continuous_pre_training=False, progress=progress)
                transfer_learning_scn03 = self.__transferLearning(transmitter_path=transmitter_path, receiver_path=receiver_path, extension='scn03', continuous_pre_training=False, progress=progress)
            transfer_learning_scnet = self.__transferLearning(transmitter_path=transmitter_path, receiver_path=receiver_path, extension='scnet', continuous_pre_training=False, progress=progress)
            if transfer_learning_scnet:
                transfer_learning_scfit = self.__transferLearning(transmitter_path=transmitter_path, receiver_path=receiver_path, extension='scfit', continuous_pre_training=False, progress=progress)
                transfer_learning_hurnet = self.__transferLearning(transmitter_path=transmitter_path, receiver_path=receiver_path, extension='hurnet', continuous_pre_training=False, progress=progress)
            return (transfer_learning_scn01 or transfer_learning_scn02 or transfer_learning_scn03) or (transfer_learning_scnet or transfer_learning_scfit or transfer_learning_hurnet)
        except Exception as error:
            if self.__show_errors: print('ERROR in transferLearning: '+str(error))
            return False
    def addFit(self, prompt='', answer='', id=0, relationship_id=0):
        try: return self.__addFit(prompt=prompt, answer=answer, id=id, relationship_id=relationship_id)
        except Exception as error:
            if self.__show_errors: print('ERROR in addFit: '+str(error))
            return False
    def fit(self, activation_function='linear', bias=0.0, learning_rate=1.0, quantization=None, method='division', progress=True):
        try: return self.__fit(activation_function=activation_function, bias=bias, learning_rate=learning_rate, quantization=quantization, method=method, progress=progress)
        except Exception as error:
            if self.__show_errors: print('ERROR in fit: '+str(error))
            return False
    def predict(self, prompt='', max_tokens=None, min_fit_probability=0.7, min_probability=0.01, generalization=True, stream=False, interval=0.0):
        try: return self.__predict(prompt=prompt, max_tokens=max_tokens, min_fit_probability=min_fit_probability, min_probability=min_probability, generalization=generalization, stream=stream, interval=interval)
        except Exception as error:
            if self.__show_errors: print('ERROR in predict: '+str(error))
            if stream == True: return self.__getStream(final_answer='', probability=0.0, interval=0.0)
            else: return {'answer': '', 'fit_probability': 0.0, 'probability': 0.0}
    def print_predict(self, prompt='', max_tokens=None, min_fit_probability=0.7, min_probability=0.01, generalization=True, stream=False, interval=0.0):
        try: self.__print_predict(prompt=prompt, max_tokens=max_tokens, min_fit_probability=min_fit_probability, min_probability=min_probability, generalization=generalization, stream=stream, interval=interval)
        except Exception as error:
            if self.__show_errors: print('ERROR in print_predict: '+str(error))
"""
This code was architected, developed, and programmed by Ben-Hur Varriano for Sapiens Technology®️ (as well as all other AI algorithms of the company),
and any unauthorized alteration, adaptation, and/or distribution, as well as public comments and/or postings regarding the operation and/or mathematics involved in the algorithm, are strictly prohibited.
Failure to comply with these rules may result in legal action against the author by our team of attorneys.

This code is an extension of the SCNet algorithm with improvements in training and inference that allow it to run on custom devices.
It features an infinite context window that makes predictions through semantic comparisons between the user prompt and the inputs of the training or adjustment samples.
The SCN network operates on a layer of the HurNetTorch network that calculates weights in a single step without backpropagation in the fine-tuning training, which allows for huge speed gains during this phase.
The data reading and processing functions work with iterative streams, which allows the use of huge data sets without memory overflow or performance loss.

We named the network SCN, an abbreviation of "Semantic Comparison Network", referring to the underlying algorithm SCNet (also authored by Ben-Hur Varriano) that originated the current code.
"""
# --------------------------> A SAPIENS TECHNOLOGY®️ PRODUCTION) <--------------------------
