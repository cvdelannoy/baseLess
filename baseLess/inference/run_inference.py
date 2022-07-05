import sys, signal, os, shutil
from pathlib import Path
from datetime import datetime
import tensorflow as tf
import numpy as np

sys.path.append(f'{list(Path(__file__).resolve().parents)[1]}')
from low_requirement_helper_functions import parse_output_path
from inference.Predictor import Predictor
from inference.ReadTable import ReadTable
from argparse_dicts import get_run_inference_parser


def main(args):

    # Limit resources
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.set_soft_device_placement(True)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if args.no_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif args.mem > 0 and len(gpus):
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [
            tf.config.experimental.VirtualDeviceConfiguration(memory_limit=args.mem)])

    # Load model
    predictor = Predictor(args.model)


    # Load read table, start table manager
    pos_reads_dir = parse_output_path(args.out_dir + 'pos_reads')
    read_table = ReadTable(args.fast5_in, pos_reads_dir, predictor)
    read_manager_process = read_table.init_table()
    abundance_array = np.zeros(predictor.nb_kmers)  # for abundance estimation mode

    # ensure processes are ended after exit
    def graceful_shutdown(sig, frame):
        print("shutting down")
        read_manager_process.terminate()
        read_manager_process.join()
        sys.exit(0)

    # Differentiate between inference modes by defining when loop stops
    if args.inference_mode == 'watch':
        signal.signal(signal.SIGINT, graceful_shutdown)
        def end_condition():
            return True
    elif args.inference_mode == 'once':
        def end_condition():
            return len(os.listdir(args.fast5_in)) != 0
    else:
        raise ValueError(f'{args.inference_mode} is not a valid inference mode')

    # Start inference loop
    start_time = datetime.now()
    while end_condition():
        read_id, read, batch_idx_list, batch_read_ids = read_table.get_reads(read_batch_size=args.batch_size)
        if read is None: continue
        pred = predictor.predict(read, batch_idx_list, batch_read_ids)
        if predictor.model_type == 'abundance':
            abundance_array += pred
            read_table.update_prediction(read_id, np.zeros(len(read_id), dtype=bool))
        elif predictor.model_type == 'read_detection':
            read_table.update_prediction(read_id, pred)
    else:
        run_time = datetime.now() - start_time
        read_manager_process.terminate()
        read_manager_process.join()
        with open(args.out_dir + 'run_stats.log', 'w') as fh:
            fh.write(f'wall_time: {run_time.seconds}s')
        print(f'wall time was {run_time.seconds} s')
        if predictor.model_type == 'abundance':
            msrd_dict = predictor.get_msrd(abundance_array)
            freq_array = abundance_array / max(abundance_array.sum(), 1)
            abundance_txt = 'kmer,abundance,frequency\n' + \
                            '\n'.join([f'{km},{ab},{fr}' for km, ab, fr in zip(predictor.kmer_list, abundance_array, freq_array)])
            msrd_txt = 'species,msrd\n' + ''.join([f'{sp},{msrd_dict[sp]}\n' for sp in msrd_dict])
            msrd_tsv = msrd_txt.replace(",", "\t")
            hit_species = min(msrd_dict, key=msrd_dict.get)
            with open(f'{args.out_dir}abundance_estimation.csv', 'w') as fh:
                fh.write(abundance_txt)
            with open(f'{args.out_dir}msrd.csv', 'w') as fh:
                fh.write(msrd_txt)
            print(f'Most likely source of reads: {hit_species}\n')
            print(f'mean squared rank difference (lower == more likely source of sample): \n {msrd_tsv}')


if __name__ == '__main__':
    parser = get_run_inference_parser()
    args = parser.parse_args()
    _ = parse_output_path(args.out_dir, clean=True)
    if args.copy_reads:
        fast5_in_new = args.out_dir + 'fast5_in/'
        shutil.copytree(args.fast5_in, fast5_in_new)
        args.fast5_in = fast5_in_new
    main(args)
