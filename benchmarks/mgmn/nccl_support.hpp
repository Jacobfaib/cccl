// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <stdexcept>
#include <string>

#include <nccl.h>

namespace mgmn
{
//! Throw on a failed NCCL call, formatting the NCCL error string.
inline void check_nccl(ncclResult_t status, const char* operation)
{
  if (status != ncclSuccess)
  {
    throw std::runtime_error(std::string{operation} + ": " + ncclGetErrorString(status));
  }
}
} // namespace mgmn
