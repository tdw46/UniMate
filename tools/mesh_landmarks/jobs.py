"""Reusable Blender modal worker lifecycle, without first-party RNA registration."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
import bpy
from . import blender_bridge

ACTIVE=set()


def runtime_python(root):
    return Path(root)/'.venv'/('Scripts/python.exe' if os.name=='nt' else 'bin/python')


def state(root):
    try:
        result=json.loads((Path(root)/'state.json').read_text())
        if not runtime_python(root).is_file():result['ready']=False
        return result
    except (OSError,ValueError):return {}


def online_allowed():
    return bool(getattr(bpy.app,'online_access',True))


def stop_process(process):
    if process.poll() is not None:return
    if os.name=='nt':
        subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True)
    else:
        try:os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError:pass
    try:process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        if os.name!='nt':
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
        else:process.kill()
        process.wait(timeout=2)


def cancel_all():
    for op in list(ACTIVE):op._cancel_requested=True


class WorkerOperator:
    """Host supplies root, mesh selection and finish_generation callback."""
    _process=None
    _timer=None
    _cancel_requested=False

    def header_text(self,seconds):
        return f'Automatic Rigging — {seconds}s · Esc to cancel'

    def start(self,context,root,action,meshes=(),device='cpu'):
        if ACTIVE:
            self.report({'WARNING'},'Another rigging job is running');return {'CANCELLED'}
        if action in {'setup','prepare'} and not online_allowed():
            self.report({'ERROR'},'Enable online access to download the rigging engine');return {'CANCELLED'}
        self._root=Path(root);self._root.mkdir(parents=True,exist_ok=True)
        self._job=Path(tempfile.mkdtemp(prefix='character-rig-'));self._action=action
        self._scene=context.scene;self._meshes=list(meshes);self._cancel_requested=False;self._binding=None
        self._log=(self._job/'worker.log').open('w');self._began=time.monotonic()
        python=runtime_python(root) if action in {'infer','verify','prepare'} else Path(sys.executable)
        cmd=[str(python),str(Path(__file__).with_name('worker.py')),action,'--root',str(root)]
        try:
            if action=='infer':
                st=state(root)
                if not st.get('ready') or not st.get('models_ready'):raise RuntimeError('Prepare the rigging engine and download its model first')
                self._digest=blender_bridge.write_input(meshes,self._job/'geometry.npz')
                cmd+=['--input',str(self._job/'geometry.npz'),'--output',str(self._job/'joints.json'),'--device',device]
            env=dict(os.environ,PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1');env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
            self._process=subprocess.Popen(cmd,stdout=self._log,stderr=subprocess.STDOUT,env=env,
                start_new_session=os.name!='nt',creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name=='nt' else 0)
            self._timer=context.window_manager.event_timer_add(.25,window=context.window)
            context.window_manager.modal_handler_add(self);ACTIVE.add(self)
            return {'RUNNING_MODAL'}
        except Exception as error:
            self.cleanup(context);self.report({'ERROR'},str(error));return {'CANCELLED'}

    def cleanup(self,context):
        if self._process:stop_process(self._process)
        if self._timer:
            try:context.window_manager.event_timer_remove(self._timer)
            except (ReferenceError,RuntimeError):pass
            self._timer=None
        if getattr(self,'_log',None):self._log.close()
        lock=self._root/'worker.lock'
        try:
            if self._process and lock.read_text()==str(self._process.pid):lock.unlink()
        except OSError:pass
        if hasattr(self,'_job'):
            # Retain only the bounded diagnostic log, never customer geometry.
            log=self._job/'worker.log'
            if log.exists():
                with log.open('rb') as f:
                    f.seek(max(0,log.stat().st_size-24000));(self._root/'last-job.log').write_bytes(f.read())
            shutil.rmtree(self._job,ignore_errors=True)
        self._binding=None
        self._prediction=None
        ACTIVE.discard(self)
        if context.area:context.area.header_text_set(None);context.area.tag_redraw()

    def modal(self,context,event):
        if event.type=='ESC' or self._cancel_requested:
            self.cleanup(context);return {'CANCELLED'}
        if event.type!='TIMER':return {'PASS_THROUGH'}
        if self._process.poll() is None:
            if context.area:context.area.header_text_set(self.header_text(int(time.monotonic()-self._began)))
            return {'PASS_THROUGH'}
        # The bind stage uses Blender's Python, not the landmark model environment.
        # Nothing in the original scene is edited until both workers succeed.
        try:
            if self._process.returncode:
                detail=(self._job/'binding-error.txt')
                raise RuntimeError(detail.read_text()[-4000:] if detail.exists() else 'Rigging worker failed. See '+str(self._root/'last-job.log'))
            if self._action in {'infer','bind'}:
                if context.scene!=self._scene or context.mode!='OBJECT':raise RuntimeError('The active scene or mode changed; generation was cancelled')
                v,f=blender_bridge.geometry(self._meshes)
                if blender_bridge.geometry_digest(v,f)!=self._digest:raise RuntimeError('Character geometry changed; generate the rig again')
            if self._action=='infer':
                result=blender_bridge.refine_prediction(self._meshes,json.loads((self._job/'joints.json').read_text()))
                options=getattr(self,'_skin_options',None)
                if options:
                    from . import skin_binding
                    command=skin_binding.prepare(self._meshes,result,self._job,options)
                    self._prediction=result
                    env=dict(os.environ,PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',RAYON_NUM_THREADS='4',BEYOND_SKINNING_BACKEND=options['backend'])
                    env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
                    self._process=subprocess.Popen(command,stdout=self._log,stderr=subprocess.STDOUT,env=env,
                        start_new_session=os.name!='nt',creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.CREATE_NO_WINDOW) if os.name=='nt' else 0)
                    self._action='bind'
                    return {'RUNNING_MODAL'}
                self.finish_generation(context,self._meshes,result)
            elif self._action=='bind':
                from . import skin_binding
                self._binding=skin_binding.load(self._job)
                self.finish_generation(context,self._meshes,self._prediction)
        except Exception as error:
            traceback.print_exc()
            if getattr(self, '_log', None) and not self._log.closed:
                traceback.print_exc(file=self._log);self._log.flush()
            self.cleanup(context);self.report({'ERROR'},str(error));return {'CANCELLED'}
        self.cleanup(context)
        self._binding=None
        self.report({'INFO'},'Automatic rigging task completed')
        return {'FINISHED'}

    def cancel(self,context):self.cleanup(context)
