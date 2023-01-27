from shutil import move
from pathlib import Path
from os.path import basename
from time import sleep

from baseLess.inference.GracefulKiller import GracefulKiller


class ReadManager(object):
    """
    Load reads into table, remove if prediction complete
    """

    def __init__(self, new_read_queue, pred_queue, batch_size, reads_dir, pos_reads_dir):
        self._new_read_queue = new_read_queue
        self._pred_queue = pred_queue
        self.batch_size = batch_size
        self.reads_dir = Path(reads_dir)
        self.pos_reads_dir = pos_reads_dir
        self.current_reads_list = []
        self.run()

    def run(self):
        killer = GracefulKiller()
        while not killer.kill_now:
            self.register_new_predictions()
            self.add_new_reads()

    def register_new_predictions(self):
        # retrieve new predictions
        new_predictions = []
        ii = 0
        while not ii > 10 and not self._pred_queue.empty():
            new_predictions.append(self._pred_queue.get())
            ii += 1
        if not len(new_predictions):
            return

        # register
        for fn, hit in new_predictions:
            if hit:
                move(fn, self.pos_reads_dir + basename(fn))
            else:
                Path(fn).unlink()
            self.current_reads_list.remove(fn)

    def add_new_reads(self):
        new_read_list = []
        for rn in self.reads_dir.iterdir():
            rn_str = str(rn)
            if rn_str not in self.current_reads_list: new_read_list.append(rn_str)
            if len(new_read_list) >= self.batch_size: break

        for nrl in new_read_list:
            while self._new_read_queue.qsize() >= self.batch_size: sleep(1)
            self._new_read_queue.put(nrl)
        self.current_reads_list.extend(new_read_list)
