import { solve_fk, solve_ik, orient_base, update_angles } from './kinematics'

var start = Date.now()
const links = [0, 4, 3, 2, 1]
const angles = [45, -90, 45, 20, 0]
const joints = [[0,0], [0,0], [0,0], [0,0], [0,0]]

var angles_rad: number[] = []
for (let i in angles)
    angles_rad.push(angles[i] * Math.PI/180)
var new_joints = solve_fk([...joints], [...angles_rad], links)
var length = 0
for (let i in links)
    length += links[i]

var target = [6, 6]
var new_angles = solve_ik(new_joints, [...angles_rad], links, length, target)
var target = [1, 8]
var new_angles = solve_ik(new_joints, [...new_angles], links, length, target)
var target = [5.3, 2.1]
var new_angles = solve_ik(new_joints, [...new_angles], links, length, target)
var target = [-6, 3]
var new_angles = solve_ik(new_joints, [...new_angles], links, length, target)

console.log("FABRIK: ", (Date.now() - start)/1000)

export {}