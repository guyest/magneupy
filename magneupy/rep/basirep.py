import tempfile
from sh import basireps_mac as basireps



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
    with open('Mn3Ge.fp', 'r+') as f:
        f.seek(0)
        fp = f.readlines()

