import tempfile, platform
if platform.system() in ['Linux', 'linux', 'linux2', 'linux4']:
    from sh import basireps_linux as basireps
elif platform.system() in ['macOS', 'Mac', 'darwin', 'Darwin']:
    from sh import basireps_mac as basireps
elif platform.system() in ['Windows']:
    from sh import basireps_win as basireps

def write_smb(crystal, file=None):
    smb = []
    smb.append('TITLE ' + crystal.name + '\n')
    smb.append('SPGR ' + crystal.spacegroup + '\n')
    smb.append('KVEC ' + str(crystal.magnetic.qm).strip('[]') + '\n')
    smb.append('ATOM ')
    if file is not None:
        file.writelines(smb)
    return smb

def read_smb(fname):
    with open(fname, 'r+') as f:
        smb = f.readlines()
    return smb

#open a temporary directory and perform calculations using BasIreps
with tempfile.TemporaryDirectory():
    with open('bas.smb', 'w') as f:
        f.writelines(smb)
    basireps('bas.smb')
    with open('bas.fp', 'r+') as f:
        f.seek(0)
        fp = f.readlines()

