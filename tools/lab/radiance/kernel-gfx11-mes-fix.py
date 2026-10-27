#!/usr/bin/env python3
"""Backport the gfx11 MES oversubscription-timer fix to a Linux v7.0.13 tree (RAD20).

On gfx11 with this kernel, any compute queue left waiting (an AQL barrier, a stream wait) keeps the GPU at 100% GFX
activity and ~95 W instead of ~10 W idle. AMD fixed it upstream (mainline master, the 7.4 merge window):
  6f04757cfe3b drm/amdgpu: add new mes misc api for gfx11
  590e26626106 drm/amdkfd: workaround 100% gpu usage issue for gfx11
  a6ea3f9ec21e drm/amdgpu: correct oversubscription timer interval for gfx11
  887a3ca213c8 drm/amdgpu: fix oversubscription timer for sr-iov
They do not cherry-pick onto v7.0.13 (mainline's MES code has moved on), so this applies the same edits by anchor. The
amdgpu user-queue half of the series targets code v7.0.13 does not have (graphics user queues) and is left out; KFD
compute queues are covered. Each edit fails loudly if its anchor is missing or ambiguous.

Usage: kernel-gfx11-mes-fix.py <linux v7.0.13 tree>
"""
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
amd = root / "drivers/gpu/drm/amd"


def edit(rel, old, new, after=None):
    path = amd / rel
    text = path.read_text()
    start = text.index(after) if after else 0
    if text.count(old, start) < 1:
        sys.exit(f"{rel}: anchor not found: {old[:60]!r}")
    if after is None and text.count(old) != 1:
        sys.exit(f"{rel}: anchor not unique: {old[:60]!r}")
    i = text.index(old, start)
    path.write_text(text[:i] + new + text[i + len(old):])


# 6f04757cfe3b: the MES misc op that tells MES there is work on an unmapped queue
edit("amdgpu/amdgpu_mes.h", "\tMES_MISC_OP_CHANGE_CONFIG,\n};",
     "\tMES_MISC_OP_CHANGE_CONFIG,\n\tMES_MISC_OP_NOTIFY_WORK_ON_UNMAPPED_QUEUE\n};")
edit("amdgpu/amdgpu_mes.h", "#endif /* __AMDGPU_MES_H__ */",
     "int amdgpu_mes_notify_unmap_queue(struct amdgpu_device *adev);\n\n#endif /* __AMDGPU_MES_H__ */")
edit("amdgpu/amdgpu_mes.c", "#if defined(CONFIG_DEBUG_FS)", """int amdgpu_mes_notify_unmap_queue(struct amdgpu_device *adev)
{
	struct mes_misc_op_input op_input = {0};
	int r;

	op_input.op = MES_MISC_OP_NOTIFY_WORK_ON_UNMAPPED_QUEUE;

	if (!adev->mes.funcs->misc_op) {
		dev_err(adev->dev, "mes notify unmap queue is not supported!\\n");
		r = -EINVAL;
		goto error;
	}

	amdgpu_mes_lock(&adev->mes);
	r = adev->mes.funcs->misc_op(&adev->mes, &op_input);
	amdgpu_mes_unlock(&adev->mes);
	if (r)
		dev_err(adev->dev, "failed to notify unmap queue.\\n");

error:
	return r;
}

#if defined(CONFIG_DEBUG_FS)""", after="int amdgpu_mes_update_enforce_isolation")
edit("amdgpu/mes_v11_0.c", """				input->change_config.option.limit_single_process;
		break;

	default:""", """				input->change_config.option.limit_single_process;
		break;
	case MES_MISC_OP_NOTIFY_WORK_ON_UNMAPPED_QUEUE:
		if ((mes->adev->mes.sched_version & AMDGPU_MES_VERSION_MASK) < 0x70) {
			dev_warn_once(mes->adev->dev,
				      "MES FW version must be larger than 0x70 to support notify work on unmapped queue.\\n");
			return 0;
		}
		misc_pkt.opcode = MESAPI_MISC__NOTIFY_WORK_ON_UNMAPPED_QUEUE;
		misc_pkt.queue_sch_level = AMD_PRIORITY_LEVEL_NORMAL;
		break;
	default:""")

# 590e26626106 + 887a3ca213c8: no MES firmware oversubscription timer on bare metal; KFD notifies MES instead, only
# while oversubscribed. a6ea3f9ec21e: the interval is 50 us (the firmware timer's unit), not 50 ms.
edit("amdgpu/mes_v11_0.c", "\tmes_set_hw_res_pkt.oversubscription_timer = 50;\n", """	/*
	 * Bare metal: disabled here, KFD arms a software replacement only
	 * while oversubscribed (see kfd_device_queue_manager.c). SR-IOV
	 * guests keep the firmware timer instead of the KFD-side one.
	 */
	if (amdgpu_sriov_vf(adev))
		mes_set_hw_res_pkt.oversubscription_timer = 50;
""")
kfd = "amdkfd/kfd_device_queue_manager.c"
edit(kfd, "#define USE_DEFAULT_GRACE_PERIOD 0xffffffff\n", """#define USE_DEFAULT_GRACE_PERIOD 0xffffffff

/* Interval for notifying MES of work on unmapped queues during oversubscription */
#define DQM_MES_UNMAP_NOTIFY_DELAY_US 50
""")
edit(kfd, """		dev_err(adev->dev, "MES might be in unrecoverable state, issue a GPU reset\\n");
		kfd_hws_hang(dqm);
	}

	return r;
}

static int remove_queue_mes(""", """		dev_err(adev->dev, "MES might be in unrecoverable state, issue a GPU reset\\n");
		kfd_hws_hang(dqm);
		return r;
	}

	/* GFX11: start notify timer only once when oversubscription begins (not under SR-IOV) */
	if (!amdgpu_sriov_vf(adev) &&
	    KFD_GC_VERSION(dqm->dev) >= IP_VERSION(11, 0, 0) &&
	    KFD_GC_VERSION(dqm->dev) < IP_VERSION(12, 0, 0) &&
	    dqm->active_cp_queue_count > get_cp_queues_num(dqm))
		queue_delayed_work(system_wq, &dqm->notify_unmap_work,
				   usecs_to_jiffies(DQM_MES_UNMAP_NOTIFY_DELAY_US));

	return r;
}

static int remove_queue_mes(""")
# remove_queue_mes: stop the timer once oversubscription clears (its first `return r;\n}` after the signature)
edit(kfd, "\treturn r;\n}\n", """
	/* GFX11: stop notify timer when oversubscription clears */
	if (!r &&
	    KFD_GC_VERSION(dqm->dev) >= IP_VERSION(11, 0, 0) &&
	    KFD_GC_VERSION(dqm->dev) < IP_VERSION(12, 0, 0) &&
	    dqm->active_cp_queue_count <= get_cp_queues_num(dqm))
		cancel_delayed_work(&dqm->notify_unmap_work);

	return r;
}
""", after="static int remove_queue_mes(")
edit(kfd, "struct device_queue_manager *device_queue_manager_init(struct kfd_node *dev)\n{",
     """static void mes_notify_unmap_work_handler(struct work_struct *work)
{
	struct device_queue_manager *dqm =
		container_of(work, struct device_queue_manager,
			     notify_unmap_work.work);

	amdgpu_mes_notify_unmap_queue((struct amdgpu_device *)dqm->dev->adev);

	/* Re-arm if still oversubscribed */
	if (READ_ONCE(dqm->active_cp_queue_count) > get_cp_queues_num(dqm))
		queue_delayed_work(system_wq, &dqm->notify_unmap_work,
				   usecs_to_jiffies(DQM_MES_UNMAP_NOTIFY_DELAY_US));
}

struct device_queue_manager *device_queue_manager_init(struct kfd_node *dev)
{""")
edit(kfd, "\t\tinit_waitqueue_head(&dqm->destroy_wait);\n\t\treturn dqm;",
     "\t\tinit_waitqueue_head(&dqm->destroy_wait);\n\t\tINIT_DELAYED_WORK(&dqm->notify_unmap_work,\n\t\t\t\t  mes_notify_unmap_work_handler);\n\t\treturn dqm;")
edit(kfd, "void device_queue_manager_uninit(struct device_queue_manager *dqm)\n{\n",
     "void device_queue_manager_uninit(struct device_queue_manager *dqm)\n{\n\tcancel_delayed_work_sync(&dqm->notify_unmap_work);\n")
edit("amdkfd/kfd_device_queue_manager.h", "\twait_queue_head_t\tdestroy_wait;\n",
     "\twait_queue_head_t\tdestroy_wait;\n\tstruct delayed_work\tnotify_unmap_work;\n")
print("gfx11 MES oversubscription-timer fix applied")
