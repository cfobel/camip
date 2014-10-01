#ifndef ___ANNEAL__SCHEDULE__HPP___
#define ___ANNEAL__SCHEDULE__HPP___

#include <stdint.h>
#include <algorithm>

namespace anneal {

using std::min;
using std::max;

template <typename REAL_T>
class AnnealSchedule {
public:
    typedef REAL_T value_type;

    REAL_T start_rlim_;
    REAL_T rlim_;
    REAL_T start_temperature_;
    REAL_T temperature_;
    REAL_T success_ratio_;

    AnnealSchedule() : start_rlim_(0), rlim_(0), start_temperature_(0),
                       temperature_(0), success_ratio_(1) {}
    AnnealSchedule(REAL_T start_rlim, REAL_T start_temperature=42.)
            : start_rlim_(0), rlim_(0), start_temperature_(0), temperature_(0),
              success_ratio_(1)  {
        this->init(start_rlim, start_temperature);
    }

    virtual void init(REAL_T start_rlim, REAL_T start_temperature=42.) {
        this->temperature_ = start_temperature;
        this->start_temperature_ = start_temperature;
        this->start_rlim_ = start_rlim;
        this->rlim_ = start_rlim;
        this->success_ratio_ = 1.;
    }

    virtual uint8_t get_temperature_stage() const {
        if(success_ratio_ > 0.96) {
            return 1;
        } else if(success_ratio_ > 0.8) {
            return 2;
        } else if(success_ratio_ > 0.8 or rlim_ > 1) {
            return 3;
        } else {
            return 4;
        }
    }

    virtual void update_state(REAL_T success_ratio) {
        success_ratio_ = success_ratio;
        update_temperature();
        update_rlim();
    }

    virtual void update_temperature() {
        uint8_t stage = get_temperature_stage();
        switch(stage) {
            case 1 :
                temperature_ *= 0.5;
                break;
            case 2 :
                temperature_ *= 0.9;
                break;
            case 3 :
                temperature_ *= 0.95;
                break;
            case 4 :
                temperature_ *= 0.8;
                break;
            default:
                temperature_ = 0;
                break;
        }
    }

    virtual void update_rlim() {
        rlim_ = min(rlim_ * (1 - 0.44 + success_ratio_), double(start_rlim_));
    }

    virtual REAL_T clamp_rlim(REAL_T max_rlim) {
        rlim_ = max(rlim_, max_rlim);
        return rlim_;
    }

    virtual ~AnnealSchedule() {}
};

} // namespace anneal

#endif  // #ifndef ___ANNEAL__SCHEDULE__HPP___
