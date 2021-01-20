import plac
import os


def msg(txt):
    print(f'-- {txt}')


class cd:
    """Context manager for changing the current working directory"""

    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)


class Extras:

    def __init__(self, prefix, force):
        self.prefix = os.getenv('CONDA_PREFIX') if prefix is None else prefix
        if force:
            msg('All dependencies specified will be re-installed')
        self.force = force

    def _check_exists(self, path):
        if os.path.exists(path):
            msg(f'{path} already exists')
            if self.force:
                msg('Cleaning directory ...')
                os.system(f'rm -rf {path}')
                os.makedirs(path)
                return False
            else:
                msg('Nothing to do.')
                return True
        else:
            os.makedirs(path)
            return False

    def __get_torch(self, torch_dir):
        msg('Fetching and installing libtorch package ...')

        with cd(torch_dir):
            zip_file = 'libtorch-cxx11-abi-shared-with-deps-1.7.1+cu110.zip'
            zip_url = f'https://download.pytorch.org/libtorch/cu110/libtorch-cxx11-abi-shared-with-deps-1.7.1%2Bcu110.zip'
            msg(f'Dowloading and unpacking {zip_url} ...')
            os.system(f'wget {zip_url}')
            os.system(f'unzip {zip_file}')

        msg(f'Libtorch fetched to {torch_dir}/libtorch, copying the files to default directories...')
        os.system(f'cp -r {torch_dir}/libtorch/share/* {self.prefix}/share')
        os.system(f'cp -r {torch_dir}/libtorch/lib/* {self.prefix}/lib')
        os.system(
            f'cp -r {torch_dir}/libtorch/include/* {self.prefix}/include')
        msg(f'Libtorch installed successfuly to {self.prefix}')

    def get_torch(self):
        msg('Libtorch required')
        torch_dir = os.path.join(self.prefix, 'torch')
        if not self._check_exists(torch_dir):
            self.__get_torch(torch_dir)


@plac.annotations(
    prefix=('Prefix of the path to where dependencies will be installed',
            'option', 'p', str),
    force=('Force to re-download and re-install', 'flag', 'f')
)
def main(prefix, force):

    fetcher = Extras(prefix, force)
    fetcher.get_torch()

    msg(f'Extra dependencies downloaded and installed from {fetcher.prefix}')


if __name__ == '__main__':
    plac.call(main)
