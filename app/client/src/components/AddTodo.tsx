import React, { useState } from 'react'

type Props = { 
  saveTodo: (e: React.FormEvent, formData: ITodo | any) => void 
}

const AddTodo: React.FC<Props> = ({ saveTodo }) => {
  const [formData] = useState<ITodo | {}>()


  return (
    <form className='Form' onSubmit={(e) => saveTodo(e, formData)}>
        <div>
          <label htmlFor='name'>Microservice One</label>
        </div>
        <div>
          <button disabled={formData === undefined ? true: false} >Execute</button>  
        </div>     
    </form>
    
  )
}

export default AddTodo
